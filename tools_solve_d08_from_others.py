"""Back-solve D08 extrinsic from D01-D07's already-calibrated extrinsics.

The chessboard is occluded by a pillar in D08's view for all frames. The SB
detector still returns 15 "corners" but several are fake (latched onto
the pillar's edges). Standard PnPRansac with 3-px threshold can't recover
because too many correspondences are wrong.

Strategy
--------
For each frame:
  1. Detect 5x3 chessboard in each of D01-D07.
  2. For each cam that detects: solvePnP using its known K + extri →
     world-frame pose of the board.
  3. Robustly average the 7 cams' estimates of the world board pose
     (use the one with lowest re-projection consensus, or median).
  4. From world board pose, compute world 3D positions of all 15 corners.
  5. Detect 5x3 corners in D08 (the 15 corners — some real, some fake).
  6. Loose PnPRansac on all 15 (3D, 2D) with thresh=50 px → rough D08 R,t.
  7. Project world corners with rough D08 R,t → predicted 2D for each.
  8. For each corner i: keep if |detected_2D[i] - predicted_2D[i]| < 15 px
     → only "real" (visible) corners kept.
  9. Pool kept (3D, 2D) correspondences across all frames.
 10. Tight PnPRansac on the pooled pool with thresh=3 px → final D08 R,t.
 11. Write into extri.yml in place.
"""

import argparse
import os
import shutil
import time
from glob import glob
from os.path import join

import cv2
import numpy as np


# ----- intri/extri IO -------------------------------------------------------

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


def read_extri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        s = names_node.at(i).string()
        if s == "":
            s = str(int(names_node.at(i).real()))
        names.append(s)
    extri = {}
    for cam in names:
        rvec = fs.getNode(f"R_{cam}").mat()
        R    = fs.getNode(f"Rot_{cam}").mat()
        T    = fs.getNode(f"T_{cam}").mat()
        extri[cam] = {"rvec": rvec, "R": R, "T": T}
    fs.release()
    return extri, names


def write_matrix(f, key, value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    f.write(f"{key}: !!opencv-matrix\n")
    f.write(f"  rows: {value.shape[0]}\n  cols: {value.shape[1]}\n  dt: d\n")
    data = ", ".join([f"{x:.6f}" for x in value.reshape(-1)])
    f.write(f"  data: [{data}]\n")


def write_extri(path, extri, names_order):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        shutil.copy(path, path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\nnames:\n")
        for cam in names_order:
            f.write(f'  - "{cam}"\n')
        for cam in names_order:
            write_matrix(f, f"R_{cam}",   extri[cam]["rvec"].reshape(3, 1))
            write_matrix(f, f"Rot_{cam}", extri[cam]["R"])
            write_matrix(f, f"T_{cam}",   extri[cam]["T"].reshape(3, 1))


# ----- chessboard ----------------------------------------------------------

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


def detect_5x3(gray, pattern=(5, 3)):
    """Multi-strategy detect, return corners or None."""
    pat_alt = (pattern[1], pattern[0])
    for pat in (pattern, pat_alt):
        try:
            ok, c = cv2.findChessboardCorners(
                gray, pat,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                      cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_FAST_CHECK)
        except cv2.error:
            ok = False
        if ok and c is not None:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
            return pat, c.reshape(-1, 2).astype(np.float32)
    for pat in (pattern, pat_alt):
        for flags in (cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY,
                      cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE):
            try:
                ok, c = cv2.findChessboardCornersSB(gray, pat, flags=flags)
            except cv2.error:
                ok = False
            if ok and c is not None:
                return pat, c.reshape(-1, 2).astype(np.float32)
    return None, None


# ----- world board pose ----------------------------------------------------

def compute_world_board_pose(detections_per_cam, extri, intri, grid,
                              good_cams):
    """Given per-cam detection of the same board, return (world R, world t,
    world_corners_3d, contributors).

    Method: each cam gives T_cam_from_board via solvePnP. Then
    T_world_from_board = T_world_from_cam @ T_cam_from_board.
    We take the contributor with smallest single-cam reproj err among
    the 7 cams as the world board pose estimate. (Cheaper than averaging
    rotations and still robust for our purposes.)
    """
    best = None  # (err_px, T_world_from_board, pattern, contributor_cam)
    for cam in good_cams:
        if cam not in detections_per_cam:
            continue
        pat, c2d = detections_per_cam[cam]
        K = intri[cam]["K"]; D = intri[cam]["dist"]
        obj = build_corners3d(pat, grid)
        try:
            ok, rv, tv = cv2.solvePnP(obj, c2d, K, D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            continue
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - c2d, axis=1).mean())
        if err > 3.0:
            continue
        # T_cam_from_board (4x4)
        Rcb, _ = cv2.Rodrigues(rv)
        Tcb = np.eye(4); Tcb[:3, :3] = Rcb; Tcb[:3, 3] = tv.reshape(3)
        # extri stores cam → world transform as: P_cam = R_extri @ P_world + T_extri,
        # i.e. R_extri = R_cam_from_world, T_extri = t_cam_from_world.
        # We need T_world_from_cam = inv(T_cam_from_world).
        R_cw = extri[cam]["R"]; t_cw = extri[cam]["T"].reshape(3)
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = t_wc
        T_wb = T_wc @ Tcb
        if best is None or err < best[0]:
            best = (err, T_wb, pat, cam)
    return best


# ----- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True,
                    help="<root>/<cam>/<frame>.jpg")
    ap.add_argument("--intri", required=True)
    ap.add_argument("--extri_in", required=True,
                    help="extri.yml from v4 BA (D01-D07 calibrated)")
    ap.add_argument("--target_cam", default="D08_20260325")
    ap.add_argument("--ref_cams", default="D01_20260325,D02_20260325,"
                    "D03_20260325,D04_20260325,D05_20260325,"
                    "D06_20260325,D07_20260325")
    ap.add_argument("--grid", type=float, default=0.19)
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--loose_thresh_px", type=float, default=50.0,
                    help="loose PnPRansac threshold for initial D08 pose")
    ap.add_argument("--inlier_radius_px", type=float, default=15.0,
                    help="corners with detected-vs-predicted dist < this "
                    "are considered real (not occluded fake)")
    ap.add_argument("--tight_thresh_px", type=float, default=3.0,
                    help="final PnPRansac threshold on filtered correspondences")
    ap.add_argument("--out_extri", required=True)
    args = ap.parse_args()

    intri, _ = read_intri(args.intri)
    extri, _ = read_extri(args.extri_in)
    ref_cams = [c.strip() for c in args.ref_cams.split(",")]
    target = args.target_cam

    Kt = intri[target]["K"]; Dt = intri[target]["dist"]
    print(f"target={target}, fx={Kt[0,0]:.1f}, "
          f"cx={Kt[0,2]:.1f}, cy={Kt[1,2]:.1f}")
    print(f"ref_cams={ref_cams}")

    # find frames common to all cams
    frame_sets = {}
    for cam in ref_cams + [target]:
        files = sorted(glob(join(args.image_root, cam, f"*{args.ext}")))
        frame_sets[cam] = set([os.path.splitext(os.path.basename(p))[0]
                               for p in files])
    common = set.intersection(*frame_sets.values())
    common = sorted(common)
    print(f"common frames: {len(common)}")

    # ---- per-frame loop ----
    pool_3d = []      # world 3D corners across frames (only inlier corners)
    pool_2d = []      # D08 2D corners (matching pool_3d)
    n_frames_with_world_pose = 0
    n_frames_target_detected = 0
    n_inlier_corners_kept = 0

    for fi, frame in enumerate(common):
        # 1) detect chessboard in each ref cam
        detections = {}
        for cam in ref_cams:
            p = join(args.image_root, cam, f"{frame}{args.ext}")
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pat, c2d = detect_5x3(gray, (5, 3))
            if c2d is not None:
                detections[cam] = (pat, c2d)
        if len(detections) < 2:
            continue
        # 2) compute world board pose from best of the ref cams
        best = compute_world_board_pose(detections, extri, intri, args.grid,
                                         ref_cams)
        if best is None:
            continue
        err_pnp, T_wb, pat, contributor = best
        n_frames_with_world_pose += 1
        # 3) 15 world 3D corner positions for this frame
        obj_local = build_corners3d(pat, args.grid)
        Rwb, twb = T_wb[:3, :3], T_wb[:3, 3]
        world_corners = (Rwb @ obj_local.T).T + twb.reshape(1, 3)

        # 4) detect target's 15 corners
        p_t = join(args.image_root, target, f"{frame}{args.ext}")
        img_t = cv2.imread(p_t)
        if img_t is None:
            continue
        gray_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
        pat_t, c2d_t = detect_5x3(gray_t, (5, 3))
        if c2d_t is None:
            continue
        # if pattern dim mismatch, skip (rare with our setup)
        if pat_t != pat:
            # try corner reversal to handle 180-flip
            if pat_t == (pat[1], pat[0]):
                # treat as transposed — skip for simplicity
                continue
            continue
        n_frames_target_detected += 1

        # 5) loose PnP on all 15 to bootstrap target extrinsic for this frame
        try:
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                world_corners.astype(np.float32),
                c2d_t.astype(np.float32),
                Kt, Dt,
                reprojectionError=args.loose_thresh_px,
                iterationsCount=2000,
                flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            continue
        if not ok or inliers is None:
            continue

        # 6) project world corners back to target with the loose pose
        proj, _ = cv2.projectPoints(world_corners.astype(np.float32),
                                     rv, tv, Kt, Dt)
        proj = proj.reshape(-1, 2)

        # also try REVERSED corner ordering to handle 180-flip in target
        proj_rev = proj[::-1]
        d_fwd = np.linalg.norm(proj - c2d_t, axis=1)
        d_rev = np.linalg.norm(proj_rev - c2d_t, axis=1)
        # heuristic: pick whichever ordering gives more sub-radius matches
        n_fwd = int((d_fwd < args.inlier_radius_px).sum())
        n_rev = int((d_rev < args.inlier_radius_px).sum())
        if n_rev > n_fwd:
            # swap target's detected order to match world ordering
            c2d_t = c2d_t[::-1]
            d_use = d_rev
        else:
            d_use = d_fwd
        mask = d_use < args.inlier_radius_px

        # require at least 4 visible corners in this frame to use it
        if mask.sum() < 4:
            continue

        kept_3d = world_corners[mask]
        kept_2d = c2d_t[mask]
        pool_3d.append(kept_3d)
        pool_2d.append(kept_2d)
        n_inlier_corners_kept += int(mask.sum())
        if fi % 20 == 0 or fi < 5:
            print(f"  [f={frame}] world from {contributor} (e={err_pnp:.2f}px),"
                  f" target kept {mask.sum()}/15 corners")

    print(f"\nworld board poses obtained: {n_frames_with_world_pose}/{len(common)}")
    print(f"target detected:            {n_frames_target_detected}")
    print(f"frames contributing inliers: {len(pool_3d)}")
    print(f"total visible-corner pool:   {n_inlier_corners_kept}")
    if len(pool_3d) < 3:
        print("FAILED: too few frames yielded visible corners")
        return

    # 7) tight PnPRansac on pooled correspondences
    obj_world = np.concatenate(pool_3d).astype(np.float32)
    img_xy = np.concatenate(pool_2d).astype(np.float32)
    print(f"\nrunning tight PnPRansac on {obj_world.shape[0]} correspondences,"
          f" thresh={args.tight_thresh_px}px")
    ok, rv, tv, inliers = cv2.solvePnPRansac(
        obj_world, img_xy, Kt, Dt,
        reprojectionError=args.tight_thresh_px,
        iterationsCount=10000,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None:
        print("tight PnPRansac FAILED")
        return
    print(f"inliers: {len(inliers)}/{obj_world.shape[0]}")

    # refine with non-linear least squares on inliers
    obj_in = obj_world[inliers.flatten()]
    img_in = img_xy[inliers.flatten()]
    rv, tv = cv2.solvePnPRefineLM(obj_in, img_in, Kt, Dt, rv, tv)
    proj, _ = cv2.projectPoints(obj_in, rv, tv, Kt, Dt)
    err_final = float(np.linalg.norm(proj.reshape(-1, 2) - img_in, axis=1).mean())
    print(f"refined mean reproj err = {err_final:.3f} px")

    # 8) write to extri.yml
    R, _ = cv2.Rodrigues(rv)
    extri[target] = {"rvec": rv.reshape(3, 1), "R": R, "T": tv.reshape(3, 1)}
    # preserve original ordering
    _, names = read_extri(args.extri_in)
    if target not in names:
        names.append(target)
    write_extri(args.out_extri, extri, names)
    print(f"\nwrote {args.out_extri}")


if __name__ == "__main__":
    main()
