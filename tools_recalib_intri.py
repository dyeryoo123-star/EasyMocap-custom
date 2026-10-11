"""Re-calibrate one or more cameras' intrinsics using chessboard detections.

Targets a specific frame range where the chessboard is known to be waved
in front of the camera. Uses cv2.findChessboardCornersSB for robust detection
with bbox-based false-positive filtering, then cv2.calibrateCamera.

Usage:
  python tools_recalib_intri.py \
      --image_root calibration_data/extri_data/extri_new_data/images \
      --cams D08_20260401170555,D03_20260401170555 \
      --pattern 5,3 --grid 0.19 \
      --frame_range 7400,9800 --step 5 \
      --intri_in  calibration_data/mocap_10000_10700/intri.yml \
      --intri_out calibration_data/mocap_10000_10700/intri_fixed.yml
"""

import argparse
import os
import shutil
import time
from glob import glob
from os.path import join

import cv2
import numpy as np


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


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        s = names_node.at(i).string()
        if s == "":
            s = str(int(names_node.at(i).real()))
        names.append(s)
    intri = {}
    for cam in names:
        K = fs.getNode(f"K_{cam}").mat()
        dist = fs.getNode(f"dist_{cam}").mat()
        intri[cam] = {"K": K, "dist": dist}
    fs.release()
    return intri, names


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


def build_corners3d(pattern, grid_size):
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            pts[idx, 0] = col * grid_size
            pts[idx, 1] = row * grid_size
            idx += 1
    return pts


def is_bbox_sane(corners, image_shape, min_side_px=30, max_side_frac=0.4):
    xy = corners.reshape(-1, 2)
    w = xy[:, 0].max() - xy[:, 0].min()
    h = xy[:, 1].max() - xy[:, 1].min()
    H, W = image_shape
    if w < min_side_px or h < min_side_px:
        return False
    if w > W * max_side_frac or h > H * max_side_frac:
        return False
    return True


def detect_chessboard(gray, pattern, scales=(1, 2)):
    for scale in scales:
        g = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            ret, c = cv2.findChessboardCornersSB(
                g, pattern,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY)
        except cv2.error:
            ret, c = False, None
        if ret and c is not None:
            pts = c.reshape(-1, 2).astype(np.float32)
            if scale != 1:
                pts = pts / scale
            return pts
    return None


def collect_views(cam, image_root, pattern, grid, frame_range, step,
                  scales, K_init=None, dist_init=None,
                  early_pnp_thresh_px=50.0, max_views=200):
    """Returns lists of (objpts, imgpts) accepted for calibration, plus image size."""
    cam_dir = join(image_root, cam)
    files = sorted(glob(join(cam_dir, "*.jpg")))
    if step > 1:
        files = files[::step]
    if frame_range:
        lo, hi = frame_range
        files = [p for p in files
                 if lo <= int(os.path.splitext(os.path.basename(p))[0]) <= hi]
    if not files:
        print(f"[{cam}] no images in range")
        return [], [], None
    obj_corners = build_corners3d(pattern, grid)
    obj_list, img_list = [], []
    image_size = None
    n_detected = 0
    n_bbox_drop = 0
    n_pnp_drop = 0
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts = detect_chessboard(gray, pattern, scales=scales)
        if pts is None:
            continue
        n_detected += 1
        if not is_bbox_sane(pts, image_size[::-1]):
            n_bbox_drop += 1
            continue
        # Early PnP sanity check if we have initial intrinsics
        if K_init is not None and dist_init is not None:
            try:
                ok, rv, tv = cv2.solvePnP(obj_corners, pts, K_init, dist_init,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    proj, _ = cv2.projectPoints(obj_corners, rv, tv,
                                                K_init, dist_init)
                    err = np.linalg.norm(
                        proj.reshape(-1, 2) - pts, axis=1).mean()
                    if err > early_pnp_thresh_px:
                        n_pnp_drop += 1
                        continue
                else:
                    n_pnp_drop += 1
                    continue
            except cv2.error:
                n_pnp_drop += 1
                continue
        obj_list.append(obj_corners.copy())
        img_list.append(pts)
        if len(obj_list) >= max_views:
            break
    print(f"[{cam}] scanned={len(files)}, detected={n_detected}, "
          f"bbox_drop={n_bbox_drop}, pnp_drop={n_pnp_drop}, "
          f"kept={len(obj_list)}")
    return obj_list, img_list, image_size


def calibrate_camera(obj_list, img_list, image_size, fix_principal=True,
                     fix_aspect=True, zero_tangent=False, focal_init=1500.0,
                     fix_k2=False, fix_k3=False):
    flags = 0
    if fix_principal:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if fix_aspect:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_USE_INTRINSIC_GUESS
    if zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if fix_k2:
        flags |= cv2.CALIB_FIX_K2
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    K_init = np.array([
        [focal_init, 0, image_size[0] / 2],
        [0, focal_init, image_size[1] / 2],
        [0, 0, 1]
    ])
    dist_init = np.zeros(5)
    print(f"    [debug] calibrate flags={flags} (FIX_K2={bool(flags & cv2.CALIB_FIX_K2)}, "
          f"FIX_K3={bool(flags & cv2.CALIB_FIX_K3)}, "
          f"FIX_ASPECT={bool(flags & cv2.CALIB_FIX_ASPECT_RATIO)}, "
          f"USE_GUESS={bool(flags & cv2.CALIB_USE_INTRINSIC_GUESS)})")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, image_size, K_init, dist_init, flags=flags)
    print(f"    [debug] dist shape={dist.shape}, raw={dist.reshape(-1)}")
    # Per-view reprojection error
    errs = []
    for i, (obj, img) in enumerate(zip(obj_list, img_list)):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean()
        errs.append(err)
    return K, dist, ret, np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--cams", required=True,
                    help="csv of camera folders to re-calibrate")
    ap.add_argument("--pattern", default="5,3")
    ap.add_argument("--grid", type=float, default=0.19)
    ap.add_argument("--frame_range", default=None,
                    help="inclusive 'lo,hi' on frame number, e.g. '7400,9800'")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--scales", default="1,2")
    ap.add_argument("--intri_in", required=True,
                    help="existing intri.yml; used for initial PnP filter "
                         "and for non-target cams to be copied verbatim")
    ap.add_argument("--intri_out", required=True)
    ap.add_argument("--fix_principal", action="store_true", default=True,
                    help="constrain cx,cy at image center (default ON)")
    ap.add_argument("--no_fix_principal", dest="fix_principal",
                    action="store_false")
    ap.add_argument("--skip_pnp_filter", action="store_true",
                    help="skip early PnP-based detection filter (use when "
                         "initial K is known to be wrong, e.g. for the camera "
                         "being re-calibrated)")
    ap.add_argument("--fix_aspect", action="store_true", default=True,
                    help="force fx == fy (square pixels). Default ON.")
    ap.add_argument("--no_fix_aspect", dest="fix_aspect", action="store_false")
    ap.add_argument("--zero_tangent", action="store_true", default=True,
                    help="force p1=p2=0 (no tangential distortion). Default ON.")
    ap.add_argument("--no_zero_tangent", dest="zero_tangent",
                    action="store_false")
    ap.add_argument("--focal_init", type=float, default=1500.0,
                    help="initial focal length guess (px); good ~1000-2000")
    ap.add_argument("--fix_k2", action="store_true", default=False,
                    help="freeze k2=0 (use only k1 radial distortion)")
    ap.add_argument("--fix_k3", action="store_true", default=True,
                    help="freeze k3=0 (default ON; k3 is rarely needed)")
    ap.add_argument("--no_fix_k3", dest="fix_k3", action="store_false")
    ap.add_argument("--max_views", type=int, default=200)
    ap.add_argument("--inlier_filter_px", type=float, default=2.0,
                    help="after first calibration, drop views with err > this "
                         "and re-calibrate (iterative outlier rejection)")
    args = ap.parse_args()

    pattern = tuple(int(x) for x in args.pattern.split(","))
    frame_range = (tuple(int(x) for x in args.frame_range.split(","))
                   if args.frame_range else None)
    scales = tuple(int(x) for x in args.scales.split(","))
    cams_to_fix = [c.strip() for c in args.cams.split(",") if c.strip()]

    intri_in, names = read_intri(args.intri_in)
    print(f"[init] loaded intri.yml with {len(names)} cams")
    print(f"[init] re-calibrating: {cams_to_fix}")
    print(f"[init] pattern={pattern}, grid={args.grid}, range={frame_range}, "
          f"step={args.step}")

    intri_out = {c: {"K": intri_in[c]["K"].copy(),
                     "dist": intri_in[c]["dist"].copy()}
                 for c in names}

    for cam in cams_to_fix:
        if cam not in intri_in:
            print(f"[{cam}] not found in intri_in; skipping")
            continue
        print(f"\n========== {cam} ==========")
        print(f"  before: fx={intri_in[cam]['K'][0,0]:.1f}, "
              f"fy={intri_in[cam]['K'][1,1]:.1f}")
        K_init = None if args.skip_pnp_filter else intri_in[cam]["K"]
        dist_init = None if args.skip_pnp_filter else intri_in[cam]["dist"]
        obj_list, img_list, image_size = collect_views(
            cam, args.image_root, pattern, args.grid,
            frame_range, args.step, scales,
            K_init=K_init, dist_init=dist_init,
            max_views=args.max_views)
        if len(obj_list) < 8:
            print(f"  [WARN] only {len(obj_list)} views, need >=8; skipping")
            continue
        # First pass: include all
        K, dist, rms, errs = calibrate_camera(
            obj_list, img_list, image_size,
            fix_principal=args.fix_principal,
            fix_aspect=args.fix_aspect,
            zero_tangent=args.zero_tangent,
            focal_init=args.focal_init,
            fix_k2=args.fix_k2, fix_k3=args.fix_k3)
        print(f"  pass1: rms={rms:.3f}, n_views={len(errs)}, "
              f"mean_err={errs.mean():.3f}, p90={np.percentile(errs,90):.3f}")
        # Iterative outlier rejection
        for it in range(3):
            mask = errs < args.inlier_filter_px
            if mask.sum() < 8 or mask.sum() == len(errs):
                break
            obj_in = [o for o, m in zip(obj_list, mask) if m]
            img_in = [i for i, m in zip(img_list, mask) if m]
            K, dist, rms, errs = calibrate_camera(
                obj_in, img_in, image_size,
                fix_principal=args.fix_principal,
                fix_aspect=args.fix_aspect,
                zero_tangent=args.zero_tangent,
                focal_init=args.focal_init,
                fix_k2=args.fix_k2, fix_k3=args.fix_k3)
            print(f"  pass{it+2}: rms={rms:.3f}, n_views={len(errs)}, "
                  f"mean_err={errs.mean():.3f}, p90={np.percentile(errs,90):.3f}")
            obj_list, img_list = obj_in, img_in
        print(f"  AFTER: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
              f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}, "
              f"k1={dist.reshape(-1)[0]:+.3f}, k2={dist.reshape(-1)[1]:+.3f}")
        intri_out[cam]["K"] = K
        intri_out[cam]["dist"] = dist.reshape(1, -1)

    write_intri(args.intri_out, intri_out, names)
    print(f"\n[write] {args.intri_out}")


if __name__ == "__main__":
    main()
