"""Re-search per-camera sync offsets using the BA-derived extrinsics.

For each non-reference camera, search a small offset window around its current
value. At each candidate, triangulate keypoints3d using all cameras (with this
cam shifted by the candidate offset) and measure the candidate camera's own
reprojection error. Pick the offset with the lowest median error.

Usage:
  python tools_search_sync.py \
      --root calibration_data/mocap_10000_10700 \
      --init calibration_data/mocap_sync_search/lags.json \
      --out  calibration_data/mocap_sync_search/lags_resync.json \
      --range 7400,9800 --step 5 --window 5 --iters 2
"""

import argparse
import json
import os
from glob import glob
from os.path import join

import cv2
import numpy as np

from easymocap.mytools.camera_utils import read_camera


def load_annot_2d(annot_path, n_joints=25, conf_min=0.3):
    """Read body25 2D keypoints for the first person; return (J, 3) array
    with zeroed entries below conf_min."""
    if not os.path.exists(annot_path):
        return np.zeros((n_joints, 3), dtype=np.float32)
    with open(annot_path) as f:
        d = json.load(f)
    annots = d.get('annots', d)
    if not annots:
        return np.zeros((n_joints, 3), dtype=np.float32)
    a = annots[0]
    k = np.asarray(a.get('keypoints', []), dtype=np.float32)
    if k.size == 0:
        return np.zeros((n_joints, 3), dtype=np.float32)
    if k.shape[0] != n_joints:
        out = np.zeros((n_joints, 3), dtype=np.float32)
        m = min(n_joints, k.shape[0])
        out[:m] = k[:m]
        k = out
    k[k[:, 2] < conf_min] = 0
    return k


def load_all_annots_indexed(root, cams, conf_min=0.3):
    """Returns dict cam -> list of (J, 3) ndarrays indexed by sorted file
    position (file-index). Also returns the sorted list of frame names per
    camera for debugging."""
    out_kp = {}
    out_names = {}
    for cam in cams:
        ann_dir = join(root, 'annots', cam)
        files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
        out_names[cam] = [f.replace('.json', '') for f in files]
        out_kp[cam] = [load_annot_2d(join(ann_dir, f), conf_min=conf_min)
                        for f in files]
    return out_kp, out_names


def triangulate_one_frame(kpts2d_views, Ps, min_view=2):
    """kpts2d_views: (V, J, 3); Ps: (V, 3, 4). Returns (J, 4) kp3d."""
    v = (kpts2d_views[:, :, 2] > 0).sum(axis=0)
    valid_joints = np.where(v >= min_view)[0]
    J = kpts2d_views.shape[1]
    result = np.zeros((J, 4), dtype=np.float32)
    if len(valid_joints) == 0:
        return result
    keypoints = kpts2d_views[:, valid_joints]
    conf3d = keypoints[:, :, 2].sum(axis=0) / v[valid_joints]
    P0 = Ps[None, :, 0, :]
    P1 = Ps[None, :, 1, :]
    P2 = Ps[None, :, 2, :]
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.concatenate([Au, Av], axis=1)
    try:
        u, s, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return result
    X = vh[:, -1, :]
    X = X / X[:, 3:]
    result[valid_joints, :3] = X[:, :3]
    result[valid_joints, 3] = conf3d
    return result


def project_to_cam(kp3d, P):
    """kp3d: (J, 4), P: (3, 4). Returns (J, 2) projected, and valid mask."""
    valid = kp3d[:, 3] > 0
    homo = np.concatenate([kp3d[:, :3], np.ones((kp3d.shape[0], 1))], axis=1)
    proj = (P @ homo.T).T
    z = proj[:, 2:3]
    uv = proj[:, :2] / np.maximum(z, 1e-9)
    return uv, valid & (z[:, 0] > 1e-6)


def evaluate_offset(target_cam, target_offset, current_offsets,
                    cams_ordered, annots, Ps, test_indices, conf_min=0.3):
    """Triangulate and measure target_cam's own reprojection error.
    All offsets are in file-index space."""
    offs = dict(current_offsets)
    offs[target_cam] = int(target_offset)
    errs = []
    P_target = Ps[cams_ordered.index(target_cam)]
    n_files = {cam: len(annots[cam]) for cam in cams_ordered}
    for fi in test_indices:
        kpts_views = []
        valid_views = []
        for cam in cams_ordered:
            j = fi + offs[cam]
            if 0 <= j < n_files[cam]:
                kpts_views.append(annots[cam][j])
                valid_views.append(True)
            else:
                kpts_views.append(np.zeros((25, 3), dtype=np.float32))
                valid_views.append(False)
        if sum(valid_views) < 3:
            continue
        kpts_views = np.stack(kpts_views, axis=0)
        kp3d = triangulate_one_frame(kpts_views, Ps, min_view=2)
        j_tgt = fi + offs[target_cam]
        if not (0 <= j_tgt < n_files[target_cam]):
            continue
        kp2d_tgt = annots[target_cam][j_tgt]
        uv, mask = project_to_cam(kp3d, P_target)
        m = mask & (kp2d_tgt[:, 2] > conf_min)
        if m.sum() == 0:
            continue
        d = np.linalg.norm(uv[m] - kp2d_tgt[m, :2], axis=1)
        errs.extend(d.tolist())
    if not errs:
        return float('inf'), 0
    return float(np.median(errs)), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='dataset root containing intri.yml, extri.yml, annots/')
    ap.add_argument('--init', type=str, default=None,
                    help='initial lags.json (optional). All offsets default to 0')
    ap.add_argument('--out', required=True, help='output lags.json')
    ap.add_argument('--range', type=str, default=None,
                    help='inclusive FILE-INDEX range "lo,hi" (positions in '
                         'sorted annot list). Default = entire range.')
    ap.add_argument('--step', type=int, default=5,
                    help='sample every Nth file-index in range for speed')
    ap.add_argument('--window', type=int, default=5,
                    help='+/- search window around current offset')
    ap.add_argument('--iters', type=int, default=2,
                    help='passes over all cams')
    ap.add_argument('--ref', type=str, default=None,
                    help='reference camera (offset fixed to 0). default = '
                         'init.ref if provided, else first camera')
    ap.add_argument('--conf', type=float, default=0.3,
                    help='2D keypoint confidence cutoff')
    args = ap.parse_args()

    root = args.root
    cams = sorted([c for c in os.listdir(join(root, 'annots'))
                   if os.path.isdir(join(root, 'annots', c))])
    print(f'[sync] cams: {cams}')

    # Initial offsets (in FILE-INDEX units)
    current = {c: 0 for c in cams}
    ref_cam = None
    if args.init:
        with open(args.init) as f:
            init_data = json.load(f)
        ref_cam = init_data.get('ref')
        for c, v in init_data.get('lags', {}).items():
            if c in current:
                current[c] = int(v)
    if args.ref:
        ref_cam = args.ref
    if ref_cam is None or ref_cam not in cams:
        ref_cam = cams[0]
    current[ref_cam] = 0
    print(f'[sync] ref camera: {ref_cam}')
    print(f'[sync] initial offsets (file-index units): {current}')

    cam_dict = read_camera(join(root, 'intri.yml'), join(root, 'extri.yml'))
    cam_dict.pop('basenames', None)
    Ps = np.stack([cam_dict[c]['P'] for c in cams], axis=0)

    annots, names = load_all_annots_indexed(root, cams, conf_min=args.conf)
    n_files = {c: len(annots[c]) for c in cams}
    for c in cams:
        print(f'[sync]   {c}: {n_files[c]} files (first={names[c][0]}, '
              f'last={names[c][-1]})')

    n_min = min(n_files.values())
    if args.range is not None:
        lo, hi = [int(x) for x in args.range.split(',')]
    else:
        lo, hi = 0, n_min - 1
    test_indices = list(range(lo, hi + 1, args.step))
    print(f'[sync] testing file-indices in [{lo}, {hi}], step={args.step}: '
          f'{len(test_indices)} indices')

    # Iterative coordinate descent
    for it in range(args.iters):
        print(f'\n=== iteration {it + 1}/{args.iters} ===')
        for cam in cams:
            if cam == ref_cam:
                continue
            best_err = float('inf')
            best_off = current[cam]
            best_n = 0
            cur = current[cam]
            print(f'\n[sync] {cam} (current = {cur:+d})')
            for delta in range(cur - args.window, cur + args.window + 1):
                err, n = evaluate_offset(cam, delta, current, cams,
                                          annots, Ps, test_indices,
                                          conf_min=args.conf)
                marker = ''
                if err < best_err:
                    best_err = err
                    best_off = delta
                    best_n = n
                    marker = ' <-'
                print(f'  delta={delta:+3d}: median_err={err:7.2f}px '
                      f'n={n:5d}{marker}')
            current[cam] = best_off
            print(f'[sync] {cam}: chose {best_off:+d} (err={best_err:.2f}px, '
                  f'n={best_n})')

    print(f'\n[sync] final offsets (file-index units): {current}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'ref': ref_cam, 'lags': current,
                   'units': 'file-index',
                   'range': [lo, hi], 'step': args.step,
                   'window': args.window, 'iters': args.iters}, f, indent=2)
    print(f'[sync] wrote {args.out}')


if __name__ == '__main__':
    main()
