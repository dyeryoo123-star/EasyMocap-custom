"""Refine multi-camera frame offsets from 2D keypoint PKL files.

The tool searches per-camera frame offsets that minimize reprojection error.
For a target camera and a candidate offset, it triangulates 3D keypoints from
the other cameras, projects those 3D points into the target camera, and
measures the target camera's 2D reprojection error. Lower error usually means
the target camera is closer to physical synchronization with the others.

Input layout:
  root/
    intri.yml
    extri.yml
    output_dir/
      keypoints2d/<cam>/<frame>.pkl

Each PKL is expected to contain {'keypoints': np.ndarray}, typically shaped
(1, 25, 3) from the myeasymocap ViTPose pipeline.
"""

import argparse
import json
import os
import pickle
from glob import glob
from os.path import basename, join, splitext

import numpy as np

from easymocap.mytools.camera_utils import read_camera


def load_pkl_2d(path, n_joints=25, conf_min=0.3):
    if not os.path.exists(path):
        return np.zeros((n_joints, 3), dtype=np.float32)
    with open(path, "rb") as f:
        data = pickle.load(f)
    k = np.asarray(data.get("keypoints", []), dtype=np.float32)
    if k.ndim == 3:
        k = k[0]
    if k.size == 0:
        return np.zeros((n_joints, 3), dtype=np.float32)
    out = np.zeros((n_joints, 3), dtype=np.float32)
    m = min(n_joints, k.shape[0])
    out[:m] = k[:m]
    out[out[:, 2] < conf_min] = 0
    return out


def load_all_keypoints(kp_root, cams, conf_min=0.3):
    out_kp, out_names = {}, {}
    for cam in cams:
        files = sorted(glob(join(kp_root, cam, "*.pkl")))
        out_names[cam] = [splitext(basename(f))[0] for f in files]
        out_kp[cam] = [load_pkl_2d(f, conf_min=conf_min) for f in files]
    return out_kp, out_names


def parse_joints(value):
    if not value:
        return None
    joints = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        joints.append(int(item))
    return sorted(set(joints))


def parse_cams(value):
    if not value:
        return None
    cams = []
    for item in value.split(","):
        item = item.strip()
        if item:
            cams.append(item)
    return cams


def filter_joints_inplace(annots, joints, n_joints=25):
    if joints is None:
        return
    keep = np.zeros(n_joints, dtype=bool)
    for joint in joints:
        if 0 <= joint < n_joints:
            keep[joint] = True
    for cam, frames in annots.items():
        for kp in frames:
            kp[~keep, 2] = 0


def triangulate_one_frame(kpts2d_views, Ps, min_view=2):
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
    A = np.concatenate([conf * (uP2 - P0), conf * (vP2 - P1)], axis=1)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return result
    X = vh[:, -1, :]
    X = X / np.maximum(X[:, 3:], 1e-9)
    result[valid_joints, :3] = X[:, :3]
    result[valid_joints, 3] = conf3d
    return result


def project_to_cam(kp3d, P):
    valid = kp3d[:, 3] > 0
    homo = np.concatenate([kp3d[:, :3], np.ones((kp3d.shape[0], 1))], axis=1)
    proj = (P @ homo.T).T
    z = proj[:, 2:3]
    uv = proj[:, :2] / np.maximum(z, 1e-9)
    return uv, valid & (z[:, 0] > 1e-6)


def evaluate_offset(target_cam, candidate_offset, current_offsets, cams, annots,
                    Ps, test_indices, conf_min=0.3, min_other_views=2,
                    triangulate_cams=None):
    offsets = dict(current_offsets)
    offsets[target_cam] = int(candidate_offset)
    n_files = {cam: len(annots[cam]) for cam in cams}
    target_idx = cams.index(target_cam)
    P_target = Ps[target_idx]
    if triangulate_cams is None:
        other_cams = [cam for cam in cams if cam != target_cam]
    else:
        other_cams = [cam for cam in triangulate_cams if cam != target_cam]
    other_Ps = np.stack([Ps[cams.index(cam)] for cam in other_cams], axis=0)

    errs = []
    frames_used = 0
    for fi in test_indices:
        other_views = []
        n_valid_views = 0
        for cam in other_cams:
            j = fi + offsets[cam]
            if 0 <= j < n_files[cam]:
                kp = annots[cam][j]
                if (kp[:, 2] > conf_min).sum() >= 5:
                    n_valid_views += 1
                other_views.append(kp)
            else:
                other_views.append(np.zeros((25, 3), dtype=np.float32))
        if n_valid_views < min_other_views:
            continue
        kp3d = triangulate_one_frame(np.stack(other_views, axis=0), other_Ps,
                                     min_view=min_other_views)
        j_tgt = fi + offsets[target_cam]
        if not (0 <= j_tgt < n_files[target_cam]):
            continue
        kp2d_tgt = annots[target_cam][j_tgt]
        uv, mask = project_to_cam(kp3d, P_target)
        m = mask & (kp2d_tgt[:, 2] > conf_min)
        if m.sum() < 4:
            continue
        d = np.linalg.norm(uv[m] - kp2d_tgt[m, :2], axis=1)
        errs.extend(d.tolist())
        frames_used += 1
    if not errs:
        return float("inf"), 0, 0, None
    arr = np.asarray(errs, dtype=np.float32)
    return float(np.median(arr)), int(arr.size), frames_used, float(np.percentile(arr, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing intri.yml/extri.yml")
    ap.add_argument("--kp_dir", default="output_sync_refine/keypoints2d")
    ap.add_argument("--intri", default=None, help="override intri.yml path")
    ap.add_argument("--extri", default=None, help="override extri.yml path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="D03_20260325")
    ap.add_argument("--range", default=None, help="inclusive index range, e.g. 0,1249")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min_other_views", type=int, default=2)
    ap.add_argument("--joints", default=None,
                    help="comma-separated keypoint indices used for scoring, e.g. 0,1,2,5,8,9,12")
    ap.add_argument("--triangulate_cams", default=None,
                    help="comma-separated cameras used to triangulate reference 3D; default uses all other cameras")
    args = ap.parse_args()

    kp_root = join(args.root, args.kp_dir)
    cams = sorted([d for d in os.listdir(kp_root) if os.path.isdir(join(kp_root, d))])
    if args.ref not in cams:
        raise ValueError(f"ref camera {args.ref} not found in {cams}")
    print(f"[offset] cams: {cams}")
    print(f"[offset] ref:  {args.ref}")

    intri_path = args.intri or join(args.root, "intri.yml")
    extri_path = args.extri or join(args.root, "extri.yml")
    print(f"[offset] intri: {intri_path}")
    print(f"[offset] extri: {extri_path}")
    cam_dict = read_camera(intri_path, extri_path)
    cam_dict.pop("basenames", None)
    Ps = np.stack([cam_dict[c]["P"] for c in cams], axis=0)

    annots, names = load_all_keypoints(kp_root, cams, conf_min=args.conf)
    joints = parse_joints(args.joints)
    triangulate_cams = parse_cams(args.triangulate_cams)
    if triangulate_cams is not None:
        missing = [cam for cam in triangulate_cams if cam not in cams]
        if missing:
            raise ValueError(f"triangulate_cams not found: {missing}")
        print(f"[offset] triangulate cams: {triangulate_cams}")
    filter_joints_inplace(annots, joints)
    if joints is not None:
        print(f"[offset] scoring joints: {joints}")
    n_files = {cam: len(annots[cam]) for cam in cams}
    for cam in cams:
        print(f"[offset] {cam}: {n_files[cam]} pkl files")
    n_min = min(n_files.values())

    if args.range:
        lo, hi = [int(x) for x in args.range.split(",")]
    else:
        lo, hi = 0, n_min - 1
    lo = max(lo, args.window)
    hi = min(hi, n_min - 1 - args.window)
    test_indices = list(range(lo, hi + 1, args.step))
    print(f"[offset] testing indices [{lo}, {hi}] step={args.step}: {len(test_indices)}")

    current = {cam: 0 for cam in cams}
    current[args.ref] = 0
    history = []
    for it in range(args.iters):
        print(f"\n=== iteration {it + 1}/{args.iters} ===")
        for cam in cams:
            if cam == args.ref:
                continue
            cur = current[cam]
            rows = []
            best = (float("inf"), cur, 0, 0, None)
            print(f"\n[offset] {cam} current={cur:+d}")
            for cand in range(cur - args.window, cur + args.window + 1):
                med, n_joints, n_frames, p90 = evaluate_offset(
                    cam, cand, current, cams, annots, Ps, test_indices,
                    conf_min=args.conf, min_other_views=args.min_other_views,
                    triangulate_cams=triangulate_cams)
                rows.append({
                    "offset": cand,
                    "median_px": med,
                    "p90_px": p90,
                    "n_joints": n_joints,
                    "n_frames": n_frames,
                })
                marker = ""
                if med < best[0]:
                    best = (med, cand, n_joints, n_frames, p90)
                    marker = " <-"
                p90_str = "inf" if p90 is None else f"{p90:7.2f}"
                print(f"  offset={cand:+3d}: median={med:7.2f}px p90={p90_str}px "
                      f"frames={n_frames:4d} joints={n_joints:5d}{marker}")
            current[cam] = int(best[1])
            history.append({"iteration": it + 1, "camera": cam, "rows": rows,
                            "chosen": int(best[1]), "chosen_median_px": best[0],
                            "chosen_p90_px": best[4],
                            "chosen_frames": best[3],
                            "chosen_joints": best[2]})
            print(f"[offset] choose {cam}: {best[1]:+d} median={best[0]:.2f}px")

    result = {
        "ref": args.ref,
        "lags": current,
        "units": "25fps-output-frame-index",
        "range": [lo, hi],
        "step": args.step,
        "window": args.window,
        "iters": args.iters,
        "conf": args.conf,
        "min_other_views": args.min_other_views,
        "joints": joints,
        "triangulate_cams": triangulate_cams,
        "intri": intri_path,
        "extri": extri_path,
        "history": history,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[offset] final offsets: {current}")
    print(f"[offset] wrote {args.out}")


if __name__ == "__main__":
    main()
