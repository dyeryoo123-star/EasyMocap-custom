"""End-to-end intrinsic calibration directly from a video.

For each camera, read the incal video, sample every N frames, detect a
chessboard at one of several candidate patterns, then run cv2.calibrateCamera
with iterative outlier rejection. Writes EasyMocap-format intri.yml.

Usage:
  python tools_calibrate_intri_from_video.py \\
      --videos_dir /Users/dyeryoo/Desktop/incal \\
      --cams D01,D02,D03,D04,D05,D06,D07,D08 \\
      --patterns 6x4,5x3,7x5,8x6 \\
      --grids 0.06,0.19,0.06,0.06 \\
      --sample_every 30 \\
      --max_views 80 \\
      --out_intri /Users/dyeryoo/Desktop/incal/intri_incal.yml

If a video filename does not start with the cam prefix exactly, the script
matches by prefix glob (e.g. D01_*.mp4).
"""

import argparse
import os
import shutil
import time
from glob import glob
from os.path import join

import cv2
import numpy as np


# ----- intri.yml I/O ---------------------------------------------------------
def write_matrix(f, key, value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    f.write(f"{key}: !!opencv-matrix\n")
    f.write(f"  rows: {value.shape[0]}\n")
    f.write(f"  cols: {value.shape[1]}\n")
    f.write("  dt: d\n")
    data = ", ".join([f"{x:.6f}" for x in value.reshape(-1)])
    f.write(f"  data: [{data}]\n")


def write_intri(path, intri, names_order):
    if os.path.exists(path):
        shutil.copy(path, path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for cam in names_order:
            f.write(f'  - "{cam}"\n')
        for cam in names_order:
            write_matrix(f, f"K_{cam}", intri[cam]["K"])
            write_matrix(f, f"dist_{cam}", intri[cam]["dist"])


# ----- chessboard detection --------------------------------------------------
def build_corners3d(pattern, grid):
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            pts[idx, 0] = c * grid
            pts[idx, 1] = r * grid
            idx += 1
    return pts


def is_bbox_sane(corners, image_shape, min_side_px=40, max_side_frac=0.7):
    xy = corners.reshape(-1, 2)
    w = xy[:, 0].max() - xy[:, 0].min()
    h = xy[:, 1].max() - xy[:, 1].min()
    H, W = image_shape
    if w < min_side_px or h < min_side_px:
        return False
    if w > W * max_side_frac and h > H * max_side_frac:
        return False
    return True


_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def detect_chessboard(gray, pattern):
    """Primary: classic findChessboardCorners (works on the large held
    8x5 board) + cornerSubPix. Fallback: findChessboardCornersSB."""
    # Try classic
    try:
        ret, c = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                  cv2.CALIB_CB_NORMALIZE_IMAGE |
                  cv2.CALIB_CB_FAST_CHECK)
    except cv2.error:
        ret, c = False, None
    if ret and c is not None:
        # refine to sub-pixel
        c = cv2.cornerSubPix(
            gray, c, (11, 11), (-1, -1), _SUBPIX_CRIT)
        return c.reshape(-1, 2).astype(np.float32)
    # Fallback: SB detector
    try:
        ret, c = cv2.findChessboardCornersSB(
            gray, pattern,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                  cv2.CALIB_CB_EXHAUSTIVE |
                  cv2.CALIB_CB_ACCURACY)
    except cv2.error:
        return None
    if not ret or c is None:
        return None
    return c.reshape(-1, 2).astype(np.float32)


# ----- per-cam pipeline ------------------------------------------------------
def detect_views_from_video(video_path, patterns_with_grid, sample_every,
                             max_views=80, max_per_pattern=None,
                             quick_probe_only=False, quick_probe_n=40):
    """Pass over a video sampling every `sample_every` frames. For each
    sampled frame try each candidate pattern; keep the first one that
    detects. Returns dict pattern_key -> {'obj': [...], 'img': [...]}, image_size.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_size = (W, H)

    per_pattern = {}
    for (pat, grid) in patterns_with_grid:
        per_pattern[pat] = {'obj_template': build_corners3d(pat, grid),
                            'obj': [], 'img': []}

    n_sampled = 0
    n_probe_limit = quick_probe_n if quick_probe_only else (n_total // sample_every + 1)
    f_idx = 0
    while f_idx < n_total and n_sampled < n_probe_limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (pat, _grid) in patterns_with_grid:
            pts = detect_chessboard(gray, pat)
            if pts is None:
                continue
            if not is_bbox_sane(pts, image_size[::-1]):
                continue
            per_pattern[pat]['obj'].append(per_pattern[pat]['obj_template'].copy())
            per_pattern[pat]['img'].append(pts)
            break  # only one pattern per frame
        n_sampled += 1
        f_idx += sample_every
        # stop once we have enough across patterns
        kept_max = max(len(v['img']) for v in per_pattern.values())
        if kept_max >= max_views and not quick_probe_only:
            break
    cap.release()
    return per_pattern, image_size, n_sampled


def calibrate_camera(obj_list, img_list, image_size, focal_init=1500.0,
                     fix_principal=True, fix_aspect=True,
                     zero_tangent=False, fix_k3=True):
    flags = 0
    if fix_principal:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS
    if zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    K_init = np.array([
        [focal_init, 0, image_size[0] / 2],
        [0, focal_init, image_size[1] / 2],
        [0, 0, 1]], dtype=np.float64)
    dist_init = np.zeros(5)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, image_size, K_init, dist_init, flags=flags)
    errs = []
    for i, (o, im) in enumerate(zip(obj_list, img_list)):
        proj, _ = cv2.projectPoints(o, rvecs[i], tvecs[i], K, dist)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - im, axis=1).mean())
    return K, dist, ret, np.array(errs)


def run_one_cam(video_path, patterns_with_grid, sample_every, max_views,
                inlier_thresh_px, focal_init):
    print(f'\n=== {os.path.basename(video_path)} ===')

    # Single full-video sweep; chessboard windows are unknown a priori.
    per_pat, imsize, n_sampled = detect_views_from_video(
        video_path, patterns_with_grid, sample_every,
        max_views=max_views, quick_probe_only=False)
    if per_pat is None:
        print('  ERROR: cannot open video')
        return None
    print(f'  image_size={imsize}, sampled {n_sampled} frames')
    for pat, d in per_pat.items():
        print(f'    pattern {pat[0]}x{pat[1]}: {len(d["img"])} hits')
    best_pat = max(per_pat, key=lambda p: len(per_pat[p]['img']))
    if len(per_pat[best_pat]['img']) == 0:
        print('  NO pattern detected anywhere')
        return None
    grid_for_best = dict(patterns_with_grid)[best_pat]
    print(f'  >>> using pattern {best_pat[0]}x{best_pat[1]} '
          f'(grid={grid_for_best})')
    obj_list = per_pat[best_pat]['obj']
    img_list = per_pat[best_pat]['img']
    if len(obj_list) < 12:
        print(f'  WARN: only {len(obj_list)} views — too few, skipping')
        return None

    # Step 3: first calibration
    K, dist, rms, errs = calibrate_camera(
        obj_list, img_list, imsize, focal_init=focal_init)
    print(f'  pass1: rms={rms:.3f}, mean={errs.mean():.3f}, '
          f'p90={np.percentile(errs,90):.3f}, n={len(errs)}')

    # Step 4: iterative outlier rejection
    for it in range(3):
        mask = errs < inlier_thresh_px
        if mask.sum() < 12 or mask.sum() == len(errs):
            break
        obj_list = [o for o, m in zip(obj_list, mask) if m]
        img_list = [i for i, m in zip(img_list, mask) if m]
        K, dist, rms, errs = calibrate_camera(
            obj_list, img_list, imsize, focal_init=focal_init)
        print(f'  pass{it+2}: rms={rms:.3f}, mean={errs.mean():.3f}, '
              f'p90={np.percentile(errs,90):.3f}, n={len(errs)}')

    print(f'  RESULT: fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
          f'cx={K[0,2]:.1f} cy={K[1,2]:.1f} '
          f'k1={dist.reshape(-1)[0]:+.4f} k2={dist.reshape(-1)[1]:+.4f}')

    return {
        'K': K, 'dist': dist.reshape(1, -1),
        'image_size': imsize, 'rms': float(rms),
        'mean_err': float(errs.mean()), 'p90_err': float(np.percentile(errs, 90)),
        'n_views': int(len(errs)),
        'pattern': f'{best_pat[0]}x{best_pat[1]}',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos_dir', required=True)
    ap.add_argument('--cams', required=True,
                    help='comma list of cam prefixes, e.g. D01,D02,...')
    ap.add_argument('--patterns', default='6x4,5x3,7x5,8x6',
                    help='candidate chessboard internal-corner patterns')
    ap.add_argument('--grids', default='0.06,0.19,0.06,0.06',
                    help='grid size in meters for each pattern in --patterns')
    ap.add_argument('--sample_every', type=int, default=30,
                    help='sample every N frames (60fps => 30 = 0.5s)')
    ap.add_argument('--max_views', type=int, default=80)
    ap.add_argument('--inlier_thresh_px', type=float, default=2.0)
    ap.add_argument('--focal_init', type=float, default=1500.0)
    ap.add_argument('--out_intri', required=True)
    ap.add_argument('--use_full_videoname', action='store_true',
                    help='write intri.yml with full filename (e.g. '
                         'D01_20260401170555) instead of short prefix (D01)')
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(',') if c.strip()]
    patterns = [tuple(int(x) for x in p.split('x'))
                for p in args.patterns.split(',')]
    grids = [float(g) for g in args.grids.split(',')]
    assert len(patterns) == len(grids), 'patterns and grids must align'
    patterns_with_grid = list(zip(patterns, grids))

    intri = {}
    names = []
    report = {}
    for cam in cams:
        matches = sorted(glob(join(args.videos_dir, f'{cam}_*.mp4'))) \
                  + sorted(glob(join(args.videos_dir, f'{cam}.mp4')))
        if not matches:
            print(f'[WARN] no video for cam {cam} in {args.videos_dir}')
            continue
        video = matches[0]
        key = (os.path.splitext(os.path.basename(video))[0]
               if args.use_full_videoname else cam)
        res = run_one_cam(video, patterns_with_grid,
                          args.sample_every, args.max_views,
                          args.inlier_thresh_px, args.focal_init)
        if res is None:
            continue
        intri[key] = {'K': res['K'], 'dist': res['dist']}
        names.append(key)
        report[key] = {
            'rms': res['rms'], 'mean_err': res['mean_err'],
            'p90_err': res['p90_err'], 'n_views': res['n_views'],
            'pattern': res['pattern'], 'image_size': res['image_size'],
        }

    os.makedirs(os.path.dirname(args.out_intri) or '.', exist_ok=True)
    write_intri(args.out_intri, intri, names)
    print(f'\n[write] {args.out_intri}')

    print('\n========== SUMMARY ==========')
    print(f'{"cam":<32}{"pattern":>10}{"n":>6}{"rms":>10}{"mean":>10}{"p90":>10}{"size":>14}')
    for cam, r in report.items():
        print(f'{cam:<32}{r["pattern"]:>10}{r["n_views"]:>6}'
              f'{r["rms"]:>10.3f}{r["mean_err"]:>10.3f}{r["p90_err"]:>10.3f}'
              f'{str(r["image_size"]):>14}')


if __name__ == '__main__':
    main()
