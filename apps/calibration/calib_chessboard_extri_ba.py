"""Multi-frame chessboard extrinsic calibration with Bundle Adjustment.

Uses the robust SB (Sector-Based) chessboard detector, multi-pattern fallback,
multi-scale, and adaptive contrast preprocessing to maximize detection coverage
across multiple cameras simultaneously, then runs global BA over per-frame
board poses + per-camera extrinsics.

This is the chessboard counterpart of calib_charuco_extri_ba.py. The BA core
is identical; only the front-end detection changes.

Run:
  python apps/calibration/calib_chessboard_extri_ba.py <data_path> \\
      --intri <intri.yml> \\
      --pattern 7,4 \\
      --grid 0.1 \\
      --step 5 \\
      --out <out_dir>
"""

import argparse
import json
import os
import shutil
import time
from glob import glob
from os.path import join

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ----------------------------- IO -----------------------------------------

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


def write_extri(path, extri, cams_order):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy(path, path + f".bak_{ts}")
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for cam in cams_order:
            f.write(f'  - "{cam}"\n')
        for cam in cams_order:
            rvec = extri[cam]["rvec"].reshape(3, 1)
            R = extri[cam]["R"]
            T = extri[cam]["T"].reshape(3, 1)
            write_matrix(f, f"R_{cam}", rvec)
            write_matrix(f, f"Rot_{cam}", R)
            write_matrix(f, f"T_{cam}", T)


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
    return intri


def list_cameras(image_root):
    return sorted([c for c in os.listdir(image_root)
                   if os.path.isdir(join(image_root, c))])


# ----------------------- BOARD GEOMETRY -----------------------------------

def build_corners3d(pattern, grid_size):
    """Pattern is OpenCV's patternSize = (cols, rows) of inner corners.
    cv2.findChessboardCorners[SB] returns corners ordered row-major
    (left-to-right within a row, then row-by-row), so the 3D model must
    follow that same ordering."""
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            pts[idx, 0] = col * grid_size
            pts[idx, 1] = row * grid_size
            idx += 1
    return pts


# ------------------------- DETECTION --------------------------------------

def preprocess_for_detection(gray, mode):
    if mode == "none":
        return gray
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if mode == "normalize":
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def detect_chessboard_robust(gray, patterns, scales, preproc_modes):
    """Try (preproc, scale, pattern, method) combos until a detection succeeds.
    Returns (pattern_used, corners2d_native_image_coords) or (None, None)."""
    for preproc in preproc_modes:
        g0 = preprocess_for_detection(gray, preproc)
        for scale in scales:
            g = g0 if scale == 1.0 else cv2.resize(
                g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for pattern in patterns:
                # Try SB first (more robust, includes subpix already)
                try:
                    retSB, corners = cv2.findChessboardCornersSB(
                        g, pattern,
                        flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                              cv2.CALIB_CB_EXHAUSTIVE |
                              cv2.CALIB_CB_ACCURACY)
                except cv2.error:
                    retSB, corners = False, None
                if retSB and corners is not None:
                    pts = corners.reshape(-1, 2).astype(np.float32)
                    if scale != 1.0:
                        pts = pts / scale
                    return pattern, pts
                # Fall back to classic
                ret, corners = cv2.findChessboardCorners(
                    g, pattern,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                          cv2.CALIB_CB_NORMALIZE_IMAGE |
                          cv2.CALIB_CB_FAST_CHECK)
                if ret:
                    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30, 0.001)
                    corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), crit)
                    pts = corners.reshape(-1, 2).astype(np.float32)
                    if scale != 1.0:
                        pts = pts / scale
                    return pattern, pts
    return None, None


def detect_all(image_root, cams, ext, patterns, scales, preproc_modes,
               step, frames_filter, verbose=True):
    """Returns dict cam -> {frame: (pattern, corners2d)}."""
    out = {c: {} for c in cams}
    for cam in cams:
        imgs = sorted(glob(join(image_root, cam, f"*{ext}")))
        if step > 1:
            imgs = imgs[::step]
        if frames_filter is not None:
            imgs = [p for p in imgs
                    if os.path.splitext(os.path.basename(p))[0] in frames_filter]
        n_ok = 0
        from time import time as _t
        t0 = _t()
        for p in imgs:
            frame = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pattern, corners = detect_chessboard_robust(
                gray, patterns, scales, preproc_modes)
            if corners is None:
                continue
            out[cam][frame] = (pattern, corners)
            n_ok += 1
        dt = _t() - t0
        if verbose:
            print(f"[detect] {cam}: scanned={len(imgs)}, valid={n_ok}, "
                  f"time={dt:.1f}s")
    return out


# ----------------------- POSE / GEOMETRY UTILS ----------------------------

def pose_to_mat(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def mat_to_rt(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3), T[:3, 3].reshape(3)


def is_geometrically_valid(corners, pattern, min_side_px=20, max_side_frac=0.95,
                           image_shape=(1080, 1920)):
    """Reject obvious false positives based on bbox sanity."""
    xy = corners.reshape(-1, 2)
    w = xy[:, 0].max() - xy[:, 0].min()
    h = xy[:, 1].max() - xy[:, 1].min()
    H, W = image_shape
    if w < min_side_px or h < min_side_px:
        return False
    if w > W * max_side_frac or h > H * max_side_frac:
        return False
    # Check the rough aspect ratio matches pattern (allow 3x off due to perspective)
    long_, short_ = pattern
    if long_ * short_ < 6:
        return True
    expected_ratio = long_ / short_
    got_ratio = max(w, h) / max(min(w, h), 1.0)
    # generous: pattern aspect should be in [1/3x, 3x] of detection aspect
    if got_ratio > expected_ratio * 4 or got_ratio < expected_ratio / 4:
        return False
    return True


def solve_pnp_per_view(detections, intri, grid_size, max_init_err_px=3.0):
    """Returns pnp[cam][frame] = (T_cam_from_board 4x4, pattern, n_pts).
    Drops solutions where initial reprojection error > max_init_err_px."""
    pnp = {c: {} for c in detections}
    drop_geom = drop_pnp = drop_err = 0
    for cam in detections:
        K = intri[cam]["K"]
        dist = intri[cam]["dist"]
        for frame, (pattern, corners) in detections[cam].items():
            if not is_geometrically_valid(corners, pattern):
                drop_geom += 1
                continue
            obj = build_corners3d(pattern, grid_size)
            img = corners.astype(np.float32)
            try:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                ok = False
            if not ok:
                drop_pnp += 1
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean()
            if err > max_init_err_px:
                drop_err += 1
                continue
            pnp[cam][frame] = (pose_to_mat(rvec, tvec), pattern, int(obj.shape[0]))
    print(f"[pnp filter] dropped by: geom={drop_geom}, pnp_fail={drop_pnp}, "
          f"err>{max_init_err_px}px={drop_err}")
    return pnp


def average_rotations(Rs):
    """Quaternion average. Rs: list of 3x3."""
    Q = []
    for R in Rs:
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
            if i == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        q = np.array([w, x, y, z])
        if len(Q) > 0 and np.dot(Q[0], q) < 0:
            q = -q
        Q.append(q)
    Q = np.stack(Q)
    M = Q.T @ Q
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, -1]
    w, x, y, z = q_avg
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    return R


def choose_ref_cam(pnp, prefer=None):
    counts = {c: len(pnp[c]) for c in pnp}
    if prefer is not None and prefer in counts and counts[prefer] > 0:
        return prefer
    return max(counts, key=lambda c: counts[c])


def init_cam_poses(pnp, ref_cam):
    """Initialize each camera's pose in the world frame (= ref camera frame)."""
    cam_T = {ref_cam: np.eye(4)}
    ref_frames = set(pnp[ref_cam].keys())
    # First pass: direct shared frames with ref
    for cam in pnp:
        if cam == ref_cam:
            continue
        common = ref_frames & set(pnp[cam].keys())
        if not common:
            continue
        Rs, Ts = [], []
        for f in common:
            T_c_b = pnp[cam][f][0]
            T_r_b = pnp[ref_cam][f][0]
            T_c_r = T_c_b @ np.linalg.inv(T_r_b)
            Rs.append(T_c_r[:3, :3])
            Ts.append(T_c_r[:3, 3])
        R_avg = average_rotations(Rs)
        T_avg = np.median(np.stack(Ts), axis=0)
        T_full = np.eye(4)
        T_full[:3, :3] = R_avg
        T_full[:3, 3] = T_avg
        cam_T[cam] = T_full
        print(f"[init] {cam}: direct via ref over {len(Rs)} frames")
    # Second pass: chain through any already-known cam
    changed = True
    while changed:
        changed = False
        for cam in pnp:
            if cam in cam_T:
                continue
            for via in list(cam_T.keys()):
                if via == cam:
                    continue
                common = set(pnp[cam].keys()) & set(pnp[via].keys())
                if not common:
                    continue
                Rs, Ts = [], []
                for f in common:
                    T_c_b = pnp[cam][f][0]
                    T_v_b = pnp[via][f][0]
                    T_c_v = T_c_b @ np.linalg.inv(T_v_b)
                    T_c_w = T_c_v @ cam_T[via]
                    Rs.append(T_c_w[:3, :3])
                    Ts.append(T_c_w[:3, 3])
                R_avg = average_rotations(Rs)
                T_avg = np.median(np.stack(Ts), axis=0)
                T_full = np.eye(4)
                T_full[:3, :3] = R_avg
                T_full[:3, 3] = T_avg
                cam_T[cam] = T_full
                print(f"[init] {cam}: chained via {via} over {len(Rs)} frames")
                changed = True
                break
    return cam_T


def init_board_poses(pnp, ref_cam, cam_T, used_frames):
    """T_{world<-board}^f for each frame. World = ref camera frame, so for
    frames where ref sees the board, T_{world<-board} is exactly the PnP
    solution from ref (which is T_{ref<-board} by construction)."""
    board_T = {}
    for f in used_frames:
        if f in pnp[ref_cam]:
            board_T[f] = pnp[ref_cam][f][0]
            continue
        for cam in pnp:
            if cam == ref_cam or cam not in cam_T:
                continue
            if f not in pnp[cam]:
                continue
            T_c_b = pnp[cam][f][0]              # cam-from-board
            T_c_w = cam_T[cam]                  # cam-from-world
            T_w_b = np.linalg.inv(T_c_w) @ T_c_b
            board_T[f] = T_w_b
            break
    return board_T


# ---------------------------- BUNDLE ADJUSTMENT ---------------------------

def build_observations(detections, pnp, used_cams, used_frames):
    """List of (cam_idx, frame_idx, pattern_used, corners2d). Only includes
    (cam, frame) pairs that survived the PnP filter — raw detections that
    failed PnP would inject bad data into the BA."""
    obs = []
    frame_idx = {f: i for i, f in enumerate(used_frames)}
    for ci, cam in enumerate(used_cams):
        for f in detections[cam]:
            if f not in frame_idx:
                continue
            if f not in pnp[cam]:
                continue
            pattern, corners = detections[cam][f]
            obs.append((ci, frame_idx[f], pattern, corners))
    return obs


def pack_params(cam_Ts, board_Ts, used_cams, used_frames, ref_idx):
    parts = []
    for i, cam in enumerate(used_cams):
        if i == ref_idx:
            continue
        r, t = mat_to_rt(cam_Ts[cam])
        parts.append(r)
        parts.append(t)
    for f in used_frames:
        r, t = mat_to_rt(board_Ts[f])
        parts.append(r)
        parts.append(t)
    return np.concatenate(parts)


def unpack_params(x, n_cams, n_frames, ref_idx):
    cam_RT = [None] * n_cams
    cam_RT[ref_idx] = (np.zeros(3), np.zeros(3))
    o = 0
    for i in range(n_cams):
        if i == ref_idx:
            continue
        cam_RT[i] = (x[o:o+3], x[o+3:o+6])
        o += 6
    board_RT = []
    for _ in range(n_frames):
        board_RT.append((x[o:o+3], x[o+3:o+6]))
        o += 6
    return cam_RT, board_RT


def project_through_board(P3_local, rvec_c, tvec_c, rvec_b, tvec_b, K, dist):
    Rb, _ = cv2.Rodrigues(rvec_b.reshape(3, 1))
    Pw = (Rb @ P3_local.T).T + tvec_b.reshape(1, 3)
    Rc, _ = cv2.Rodrigues(rvec_c.reshape(3, 1))
    Pc = (Rc @ Pw.T).T + tvec_c.reshape(1, 3)
    proj, _ = cv2.projectPoints(Pc.astype(np.float32),
                                np.zeros(3), np.zeros(3), K, dist)
    return proj.reshape(-1, 2)


def run_ba(observations, used_cams, used_frames, grid_size,
           intri, cam_Ts, board_Ts, ref_cam,
           huber_pixels=2.0, max_nfev=120, verbose=2):
    ref_idx = used_cams.index(ref_cam)
    n_cams = len(used_cams)
    n_frames = len(used_frames)

    # Flatten: each observation contributes N corners (variable per pattern)
    # Build flat list of (ci, fi, corner3d_local, uv)
    obs_flat = []
    for (ci, fi, pattern, corners) in observations:
        P3 = build_corners3d(pattern, grid_size)
        for k in range(P3.shape[0]):
            obs_flat.append((ci, fi, P3[k], corners[k]))
    n_obs = len(obs_flat)
    print(f"[ba] cams={n_cams} (ref={ref_cam}), frames={n_frames}, "
          f"observations={n_obs}")

    Ks = [intri[c]["K"] for c in used_cams]
    Ds = [intri[c]["dist"] for c in used_cams]

    # Group by (cam, frame) for batch project
    by_cf = {}
    for oi, (ci, fi, p3, uv) in enumerate(obs_flat):
        by_cf.setdefault((ci, fi), []).append((oi, p3, uv))

    target_uv = np.array([uv for (_, _, _, uv) in obs_flat], dtype=np.float64)

    def residual(x):
        cam_RT, board_RT = unpack_params(x, n_cams, n_frames, ref_idx)
        proj_all = np.zeros((n_obs, 2))
        for (ci, fi), items in by_cf.items():
            P3 = np.array([p3 for _, p3, _ in items], dtype=np.float32)
            rc, tc = cam_RT[ci]
            rb, tb = board_RT[fi]
            proj = project_through_board(P3, rc, tc, rb, tb, Ks[ci], Ds[ci])
            for k, (oi, _, _) in enumerate(items):
                proj_all[oi] = proj[k]
        return (proj_all - target_uv).reshape(-1)

    n_params = 6 * (n_cams - 1) + 6 * n_frames
    jac = lil_matrix((2 * n_obs, n_params), dtype=np.uint8)
    for oi, (ci, fi, _, _) in enumerate(obs_flat):
        if ci != ref_idx:
            cam_offset = 6 * (ci if ci < ref_idx else ci - 1)
            for d in range(6):
                jac[2*oi, cam_offset + d] = 1
                jac[2*oi+1, cam_offset + d] = 1
        b_offset = 6 * (n_cams - 1) + 6 * fi
        for d in range(6):
            jac[2*oi, b_offset + d] = 1
            jac[2*oi+1, b_offset + d] = 1

    x0 = pack_params(cam_Ts, board_Ts, used_cams, used_frames, ref_idx)
    r0 = residual(x0)
    print(f"[ba] initial RMS = {np.sqrt(np.mean(r0**2)):.3f} px")

    result = least_squares(
        residual, x0,
        jac_sparsity=jac.tocsr(),
        method="trf",
        loss="huber",
        f_scale=huber_pixels,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=verbose,
    )
    r1 = residual(result.x)
    print(f"[ba] final RMS = {np.sqrt(np.mean(r1**2)):.3f} px")
    return result.x, obs_flat


def assemble_extri(x, used_cams, used_frames, ref_cam):
    ref_idx = used_cams.index(ref_cam)
    n_cams = len(used_cams)
    cam_RT, _ = unpack_params(x, n_cams, len(used_frames), ref_idx)
    extri = {}
    for i, cam in enumerate(used_cams):
        rvec, tvec = cam_RT[i]
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        extri[cam] = {"rvec": rvec.reshape(3, 1),
                      "R": R,
                      "T": tvec.reshape(3, 1)}
    return extri


def report_per_cam_error(obs_flat, used_cams, used_frames, intri, x, ref_cam):
    ref_idx = used_cams.index(ref_cam)
    n_cams = len(used_cams)
    n_frames = len(used_frames)
    cam_RT, board_RT = unpack_params(x, n_cams, n_frames, ref_idx)

    per = {c: [] for c in used_cams}
    by_cf = {}
    for (ci, fi, p3, uv) in obs_flat:
        by_cf.setdefault((ci, fi), []).append((p3, uv))
    for (ci, fi), items in by_cf.items():
        cam = used_cams[ci]
        P3 = np.array([p3 for p3, _ in items], dtype=np.float32)
        uvs = np.array([uv for _, uv in items], dtype=np.float32)
        rc, tc = cam_RT[ci]
        rb, tb = board_RT[fi]
        proj = project_through_board(P3, rc, tc, rb, tb,
                                     intri[cam]["K"], intri[cam]["dist"])
        err = np.linalg.norm(proj - uvs, axis=1)
        per[cam].extend(err.tolist())

    print("\n[report] per-camera reprojection error after BA:")
    print(f"  {'cam':<25} {'n':>6} {'mean':>8} {'median':>8} "
          f"{'p90':>8} {'p95':>8} {'p99':>8}")
    summary = {}
    for cam in used_cams:
        e = np.array(per[cam])
        if e.size == 0:
            print(f"  {cam:<25} {0:>6} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            summary[cam] = None
            continue
        print(f"  {cam:<25} {e.size:>6d} {e.mean():>8.3f} "
              f"{np.median(e):>8.3f} {np.percentile(e,90):>8.3f} "
              f"{np.percentile(e,95):>8.3f} {np.percentile(e,99):>8.3f}")
        summary[cam] = {"n": int(e.size), "mean": float(e.mean()),
                       "median": float(np.median(e)),
                       "p95": float(np.percentile(e, 95))}
    return summary


# --------------------------- COVERAGE REPORT ------------------------------

def rescue_camera(rescue_cam, used_frames, world_board_Ts, intri,
                  image_root, ext, patterns, scales, preproc_modes, grid,
                  ransac_thresh_px=3.0, per_view_max_err_px=50.0):
    """Use already-converged world board poses to bootstrap a camera that was
    dropped from BA. Runs aggressive detection on the rescue cam, builds
    (3D_world, 2D_cam) correspondences (testing both corner orderings per
    frame), then PnPRansac for a single robust extrinsic."""
    K = intri[rescue_cam]["K"]
    dist = intri[rescue_cam]["dist"]

    n_detected = 0
    pool_3d = []
    pool_2d = []
    per_frame_err = []
    for f in used_frames:
        if f not in world_board_Ts:
            continue
        p = join(image_root, rescue_cam, f"{f}{ext}")
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pattern, corners = detect_chessboard_robust(
            gray, patterns, scales, preproc_modes)
        if corners is None:
            continue
        n_detected += 1
        # World 3D points for this frame's board pose
        P_local = build_corners3d(pattern, grid)
        T_w_b = world_board_Ts[f]
        Rwb, twb = T_w_b[:3, :3], T_w_b[:3, 3]
        P_world = (Rwb @ P_local.T).T + twb.reshape(1, 3)
        # Try both orderings, score with single-frame PnP reproj
        def score(corners_v):
            ok, rv, tv = cv2.solvePnP(P_world.astype(np.float32),
                                      corners_v.astype(np.float32),
                                      K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return np.inf
            proj, _ = cv2.projectPoints(P_world.astype(np.float32),
                                        rv, tv, K, dist)
            return float(np.linalg.norm(
                proj.reshape(-1, 2) - corners_v, axis=1).mean())
        err_fwd = score(corners)
        err_rev = score(corners[::-1])
        best_err = min(err_fwd, err_rev)
        if best_err > per_view_max_err_px:
            continue
        chosen = corners if err_fwd <= err_rev else corners[::-1]
        pool_3d.append(P_world)
        pool_2d.append(chosen)
        per_frame_err.append(best_err)

    print(f"[rescue {rescue_cam}] raw detections: {n_detected}, "
          f"usable frames (per-frame err < {per_view_max_err_px}px): "
          f"{len(pool_3d)}")
    if len(pool_3d) < 1:
        print(f"[rescue {rescue_cam}] FAILED: no usable frames")
        return None

    obj = np.concatenate([p.astype(np.float32) for p in pool_3d])
    img2d = np.concatenate([p.astype(np.float32) for p in pool_2d])
    print(f"[rescue {rescue_cam}] PnPRansac on {obj.shape[0]} correspondences "
          f"(thresh={ransac_thresh_px}px)")
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img2d, K, dist,
            reprojectionError=ransac_thresh_px,
            iterationsCount=500,
            flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error as e:
        print(f"[rescue {rescue_cam}] solvePnPRansac error: {e}")
        return None
    if not ok or inliers is None or len(inliers) < 6:
        print(f"[rescue {rescue_cam}] FAILED: ransac found too few inliers")
        return None
    inl = inliers.reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(obj[inl], img2d[inl], K, dist,
                                  rvec, tvec, True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img2d[inl], axis=1).mean())
    print(f"[rescue {rescue_cam}] RANSAC inliers {len(inl)}/{obj.shape[0]}, "
          f"refined mean err = {err:.2f}px")
    R, _ = cv2.Rodrigues(rvec)
    return {"rvec": rvec, "R": R, "T": tvec,
            "n_inliers": int(len(inl)), "n_total": int(obj.shape[0]),
            "mean_err_px": err}


def coverage_report(pnp, cams):
    by_frame = {}
    for cam in cams:
        for f in pnp[cam]:
            by_frame.setdefault(f, set()).add(cam)
    counts = [len(v) for v in by_frame.values()]
    print(f"\n[coverage] {len(by_frame)} frames have >=1 cam detection")
    for k in [2, 3, 4, 5, 6, 7, 8]:
        n = sum(1 for c in counts if c >= k)
        print(f"  >= {k} cams: {n} frames")
    # Pair coverage matrix
    print("[coverage] pair overlap (frames seen by both):")
    print("  " + " ".join([c[:3] for c in cams]))
    for c1 in cams:
        row = [c1[:3] + ":"]
        s1 = set(pnp[c1].keys())
        for c2 in cams:
            s2 = set(pnp[c2].keys())
            row.append(f"{len(s1 & s2):>4d}")
        print("  ".join(row))


# --------------------------------- MAIN -----------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Multi-frame chessboard extri with bundle adjustment")
    p.add_argument("path", type=str, help="dataset root (images/<cam>/*.jpg)")
    p.add_argument("--image", type=str, default="images")
    p.add_argument("--ext", type=str, default=".jpg")
    p.add_argument("--out", type=str, default=None,
                   help="output dir; default <path>/output_chessboard_ba")
    p.add_argument("--intri", type=str, required=True)
    p.add_argument("--cams", type=str, default=None,
                   help="csv of cameras; default = all under images/")
    p.add_argument("--ref_cam", type=str, default=None)
    # board
    p.add_argument("--pattern", type=str, default="7,4",
                   help="primary pattern as 'long,short' (inner corners)")
    p.add_argument("--extra_patterns", type=str, default="",
                   help="csv of fallback patterns, e.g. '5,3:9,6'")
    p.add_argument("--grid", type=float, default=0.19,
                   help="grid (square) size in meters")
    # detection
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--frames", type=str, default=None)
    p.add_argument("--frame_range", type=str, default=None,
                   help="inclusive integer range 'start,end' on frame name (e.g. '7400,9800')")
    p.add_argument("--scales", type=str, default="1,2",
                   help="multi-scale list, e.g. '1,2'")
    p.add_argument("--preproc", type=str, default="none,clahe",
                   help="csv of preproc modes from {none,clahe,normalize}")
    p.add_argument("--max_init_err_px", type=float, default=5.0)
    # BA
    p.add_argument("--huber_px", type=float, default=2.0)
    p.add_argument("--max_nfev", type=int, default=120)
    p.add_argument("--max_frames", type=int, default=400)
    p.add_argument("--outlier_view_px", type=float, default=5.0,
                   help="stage-2 cutoff: drop (cam,frame) views whose post-BA "
                        "median per-corner err exceeds this")
    # rescue
    p.add_argument("--rescue_cams", type=str, default="",
                   help="csv of cameras to force-rescue using BA-derived 3D")
    p.add_argument("--min_rescue_frames", type=int, default=5,
                   help="auto-rescue any cam with fewer than this many views")
    p.add_argument("--rescue_ransac_px", type=float, default=3.0,
                   help="PnPRansac reprojection threshold in pixels")
    # output control
    p.add_argument("--coverage_only", action="store_true",
                   help="run detection + report coverage, then exit")
    args = p.parse_args()

    # Parse pattern(s)
    def parse_pat(s):
        a, b = s.split(",")
        return (int(a), int(b))
    primary = parse_pat(args.pattern)
    extras = []
    if args.extra_patterns.strip():
        for tok in args.extra_patterns.split(":"):
            if tok.strip():
                extras.append(parse_pat(tok))
    patterns = [primary] + [p for p in extras if p != primary]
    print(f"[board] patterns to try: {patterns}, grid_size={args.grid}")

    image_root = join(args.path, args.image)
    all_cams = list_cameras(image_root)
    if args.cams is not None:
        wanted = [c.strip() for c in args.cams.split(",") if c.strip()]
        used_cams = [c for c in wanted if c in all_cams]
    else:
        used_cams = all_cams
    print(f"[cams] {len(used_cams)} cameras: {used_cams}")

    intri = read_intri(args.intri)
    missing = [c for c in used_cams if c not in intri]
    if missing:
        raise RuntimeError(f"intri.yml missing cameras: {missing}")

    frames_filter = None
    if args.frames is not None:
        frames_filter = set([x.strip() for x in args.frames.split(",") if x.strip()])
    elif args.frame_range is not None:
        lo, hi = [int(x) for x in args.frame_range.split(",")]
        # Match by integer value of frame name, after stripping leading zeros
        frames_filter = {f"{i:06d}" for i in range(lo, hi + 1)}
        print(f"[frame_range] using frames {lo}..{hi} ({len(frames_filter)} candidate names)")

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    preproc_modes = [x.strip() for x in args.preproc.split(",") if x.strip()]
    print(f"[detect] scales={scales}, preproc_modes={preproc_modes}, step={args.step}")

    detections = detect_all(image_root, used_cams, args.ext,
                            patterns, scales, preproc_modes,
                            args.step, frames_filter)

    total = sum(len(detections[c]) for c in used_cams)
    if total == 0:
        raise RuntimeError("No chessboard detections at all.")

    # Per-view PnP (also filters bad ones)
    pnp = solve_pnp_per_view(detections, intri, args.grid,
                             max_init_err_px=args.max_init_err_px)
    for c in used_cams:
        print(f"[pnp] {c}: {len(pnp[c])} usable frames")

    coverage_report(pnp, used_cams)

    if args.coverage_only:
        print("\n[done] coverage-only run, exiting before BA.")
        return

    ref_cam = choose_ref_cam(pnp, prefer=args.ref_cam)
    print(f"[ref] reference camera = {ref_cam}")

    cam_Ts = init_cam_poses(pnp, ref_cam)
    missing = [c for c in used_cams if c not in cam_Ts]
    if missing:
        print(f"[init] dropping cameras with no path to ref: {missing}")
        used_cams = [c for c in used_cams if c in cam_Ts]

    # Frames seen by >=2 used cams
    frame_count = {}
    for cam in used_cams:
        for f in pnp[cam]:
            frame_count[f] = frame_count.get(f, 0) + 1
    used_frames = sorted([f for f, n in frame_count.items() if n >= 2])
    if not used_frames:
        raise RuntimeError("No frame is seen by >= 2 cameras with valid PnP.")
    if len(used_frames) > args.max_frames:
        idx = np.linspace(0, len(used_frames) - 1, args.max_frames, dtype=int)
        used_frames = [used_frames[i] for i in idx]
    print(f"[frames] using {len(used_frames)} multi-view frames")

    board_Ts = init_board_poses(pnp, ref_cam, cam_Ts, used_frames)
    used_frames = [f for f in used_frames if f in board_Ts]

    observations = build_observations(detections, pnp, used_cams, used_frames)
    print(f"[obs] {len(observations)} (cam,frame) views in BA input")

    x_opt, obs_flat = run_ba(
        observations, used_cams, used_frames, args.grid,
        intri, cam_Ts, board_Ts, ref_cam,
        huber_pixels=args.huber_px, max_nfev=args.max_nfev)

    print("\n[stage1] errors before corner alignment:")
    report_per_cam_error(obs_flat, used_cams, used_frames,
                         intri, x_opt, ref_cam)

    # ---- Stage 1.5: corner-order alignment
    # OpenCV chessboard detection has an inherent 180-degree ambiguity for
    # asymmetric patterns: different cameras may number corner 0 at
    # opposite ends of the same physical board. Use stage-1's pose estimates
    # to detect this and flip the 2D corner ordering where it helps.
    ref_idx_v1 = used_cams.index(ref_cam)
    cam_RT_v1, board_RT_v1 = unpack_params(
        x_opt, len(used_cams), len(used_frames), ref_idx_v1)
    aligned_observations = []
    flipped = 0
    for (ci, fi, pattern, corners) in observations:
        cam = used_cams[ci]
        P3 = build_corners3d(pattern, args.grid)
        rc, tc = cam_RT_v1[ci]
        rb, tb = board_RT_v1[fi]
        proj = project_through_board(P3, rc, tc, rb, tb,
                                     intri[cam]["K"], intri[cam]["dist"])
        err_fwd = np.linalg.norm(proj - corners, axis=1).mean()
        err_rev = np.linalg.norm(proj - corners[::-1], axis=1).mean()
        # Flip only when reversal yields a meaningful improvement and the
        # forward order was clearly wrong (> 5px mean residual).
        if err_rev < err_fwd * 0.5 and err_fwd > 5.0:
            aligned_observations.append((ci, fi, pattern, corners[::-1].copy()))
            flipped += 1
        else:
            aligned_observations.append((ci, fi, pattern, corners))
    print(f"\n[align] flipped corner order on {flipped} of "
          f"{len(observations)} (cam,frame) views")
    observations = aligned_observations

    # ---- Stage 1.6: re-run BA with aligned corners so subsequent outlier
    # rejection sees the corrected residuals.
    if flipped > 0:
        print("[stage1.6] re-running BA after corner alignment")
        x_opt, obs_flat = run_ba(
            observations, used_cams, used_frames, args.grid,
            intri, cam_Ts, board_Ts, ref_cam,
            huber_pixels=args.huber_px, max_nfev=args.max_nfev)
        print("\n[stage1.6] errors after corner alignment:")
        report_per_cam_error(obs_flat, used_cams, used_frames,
                             intri, x_opt, ref_cam)

    # ---- Stage 2: outlier rejection per (cam, frame) view, then re-run BA
    print(f"\n[stage2] rejecting (cam,frame) views with median err > "
          f"{args.outlier_view_px:.1f}px, then re-running BA")
    ref_idx = used_cams.index(ref_cam)
    cam_RT, board_RT = unpack_params(x_opt, len(used_cams), len(used_frames), ref_idx)
    # Group by (ci, fi) → list of (p3, uv)
    by_cf = {}
    for (ci, fi, p3, uv) in obs_flat:
        by_cf.setdefault((ci, fi), []).append((p3, uv))
    keep_cf = set()
    drop_cf = 0
    for (ci, fi), items in by_cf.items():
        cam = used_cams[ci]
        P3 = np.array([p3 for p3, _ in items], dtype=np.float32)
        uvs = np.array([uv for _, uv in items], dtype=np.float32)
        rc, tc = cam_RT[ci]
        rb, tb = board_RT[fi]
        proj = project_through_board(P3, rc, tc, rb, tb,
                                     intri[cam]["K"], intri[cam]["dist"])
        err = np.linalg.norm(proj - uvs, axis=1)
        if np.median(err) <= args.outlier_view_px:
            keep_cf.add((ci, fi))
        else:
            drop_cf += 1
    print(f"[stage2] dropped {drop_cf} of {len(by_cf)} (cam,frame) views")

    # Rebuild observations from kept views
    inlier_observations = []
    for (ci, fi, pattern, corners) in observations:
        if (ci, fi) in keep_cf:
            inlier_observations.append((ci, fi, pattern, corners))

    # Drop frames now seen by < 2 cams; drop cams now seen by 0 frames
    frame_use = {fi: 0 for fi in range(len(used_frames))}
    cam_use = {ci: 0 for ci in range(len(used_cams))}
    for (ci, fi, _, _) in inlier_observations:
        frame_use[fi] += 1
        cam_use[ci] += 1
    keep_frame_idx = [fi for fi, n in frame_use.items() if n >= 2]
    keep_cam_idx = [ci for ci, n in cam_use.items() if n >= 1]
    if used_cams.index(ref_cam) not in keep_cam_idx:
        raise RuntimeError("ref cam dropped after outlier rejection")
    if not keep_frame_idx:
        raise RuntimeError("no frames left after outlier rejection")

    # Subset cams/frames keeping original order
    keep_cam_idx = sorted(keep_cam_idx)
    keep_frame_idx = sorted(keep_frame_idx)
    new_cams = [used_cams[i] for i in keep_cam_idx]
    new_frames = [used_frames[i] for i in keep_frame_idx]
    # Re-index observations
    ci_map = {old: new for new, old in enumerate(keep_cam_idx)}
    fi_map = {old: new for new, old in enumerate(keep_frame_idx)}
    inlier_observations = [(ci_map[ci], fi_map[fi], pat, cor)
                           for (ci, fi, pat, cor) in inlier_observations
                           if ci in ci_map and fi in fi_map]
    print(f"[stage2] now {len(new_cams)} cams x {len(new_frames)} frames, "
          f"{len(inlier_observations)} views")

    # Rebuild init cam_Ts and board_Ts from stage-1 params
    new_cam_Ts = {}
    for new_ci, old_ci in enumerate(keep_cam_idx):
        rvec, tvec = cam_RT[old_ci]
        new_cam_Ts[used_cams[old_ci]] = pose_to_mat(
            rvec.reshape(3, 1), tvec.reshape(3, 1))
    new_board_Ts = {}
    for new_fi, old_fi in enumerate(keep_frame_idx):
        rvec, tvec = board_RT[old_fi]
        new_board_Ts[used_frames[old_fi]] = pose_to_mat(
            rvec.reshape(3, 1), tvec.reshape(3, 1))

    x_opt2, obs_flat2 = run_ba(
        inlier_observations, new_cams, new_frames, args.grid,
        intri, new_cam_Ts, new_board_Ts, ref_cam,
        huber_pixels=args.huber_px, max_nfev=args.max_nfev)

    print("\n[final] errors after stage-2 BA:")
    summary = report_per_cam_error(obs_flat2, new_cams, new_frames,
                                   intri, x_opt2, ref_cam)
    used_cams = new_cams
    used_frames = new_frames
    x_opt = x_opt2

    # ---- Stage 3: rescue dropped/weak cameras using BA-derived world points
    ref_idx_final = used_cams.index(ref_cam)
    cam_RT_final, board_RT_final = unpack_params(
        x_opt, len(used_cams), len(used_frames), ref_idx_final)
    # World board poses (T_{world<-board}) per frame
    world_board_Ts = {}
    for fi, f in enumerate(used_frames):
        rb, tb = board_RT_final[fi]
        world_board_Ts[f] = pose_to_mat(rb.reshape(3, 1), tb.reshape(3, 1))

    # Identify rescue candidates: cams in all_cams not in used_cams,
    # OR explicitly listed via --rescue_cams,
    # OR (default) cams with < min_rescue_frames usable frames in current BA.
    rescue_candidates = set(args.rescue_cams.split(",")) if args.rescue_cams else set()
    rescue_candidates = {c.strip() for c in rescue_candidates if c.strip()}
    # Add automatic candidates
    cam_use_count = {ci: 0 for ci in range(len(used_cams))}
    for (ci, fi, _, _) in observations:
        cam_use_count[ci] += 1
    for c in all_cams:
        if c not in used_cams:
            rescue_candidates.add(c)
        else:
            ci = used_cams.index(c)
            if cam_use_count[ci] < args.min_rescue_frames:
                rescue_candidates.add(c)
    rescue_candidates.discard(ref_cam)

    rescued = {}
    for rescue_cam in sorted(rescue_candidates):
        if rescue_cam not in intri:
            print(f"[rescue] skip {rescue_cam}: no intrinsics")
            continue
        print(f"\n[rescue] {rescue_cam}")
        result = rescue_camera(
            rescue_cam, used_frames, world_board_Ts, intri,
            image_root, args.ext, [primary] + extras, scales,
            preproc_modes, args.grid,
            ransac_thresh_px=args.rescue_ransac_px)
        if result is not None:
            rescued[rescue_cam] = result

    out_dir = args.out or join(args.path, "output_chessboard_ba")
    os.makedirs(out_dir, exist_ok=True)
    extri = assemble_extri(x_opt, used_cams, used_frames, ref_cam)
    # Append rescued cameras
    final_cam_order = list(used_cams)
    for cam, res in rescued.items():
        extri[cam] = {"rvec": res["rvec"], "R": res["R"], "T": res["T"]}
        if cam not in final_cam_order:
            final_cam_order.append(cam)
    # Sort camera order to match expected (alphabetical / by name)
    final_cam_order = sorted(final_cam_order)
    extri_path = join(out_dir, "extri.yml")
    write_extri(extri_path, extri, final_cam_order)
    print(f"\n[write] extri.yml -> {extri_path}")

    intri_out = join(out_dir, "intri.yml")
    shutil.copy(args.intri, intri_out)
    print(f"[write] intri.yml -> {intri_out}")

    meta = {
        "ref_cam": ref_cam,
        "cams": used_cams,
        "rescued_cams": {
            c: {k: v for k, v in r.items()
                if k in ("n_inliers", "n_total", "mean_err_px")}
            for c, r in rescued.items()},
        "final_cam_order": final_cam_order,
        "n_frames_used": len(used_frames),
        "n_observations": int(len(obs_flat)),
        "pattern_primary": list(primary),
        "extra_patterns": [list(p) for p in extras],
        "grid": args.grid,
        "step": args.step,
        "scales": scales,
        "preproc": preproc_modes,
        "huber_px": args.huber_px,
        "per_cam_err": summary,
    }
    with open(join(out_dir, "ba_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] ba_meta.json -> {join(out_dir, 'ba_meta.json')}")


if __name__ == "__main__":
    main()
