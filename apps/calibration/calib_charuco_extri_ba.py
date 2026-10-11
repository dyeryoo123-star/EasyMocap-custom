"""Multi-frame Charuco extrinsic calibration with Bundle Adjustment.

Pipeline:
  1. Detect Charuco corners across many frames per camera.
  2. Per (cam, frame) cv2.solvePnP -> board pose in camera frame.
  3. Choose reference camera (most successful PnP); world = ref camera frame.
  4. Robust initialization of every other camera's pose via cross-frame averaging.
  5. Initialize each used frame's board-pose in world from ref cam PnP.
  6. Bundle-adjust {non-ref cam poses} + {per-frame board poses} jointly,
     minimizing reprojection error with Huber loss on all (cam, frame, corner)
     observations, using scipy.optimize.least_squares with sparse Jacobian.
  7. Write extri.yml in EasyMocap format. Backs up any existing file.

Run:
  python apps/calibration/calib_charuco_extri_ba.py <data_path> \\
      --intri <intri.yml> \\
      --template lab_charuco_12x8 \\
      --id_layout_json <id_layout.json> \\
      --step 10 \\
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


ARUCO_DICT = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
}

BOARD_TEMPLATES = {
    "easymocap_default": {"long": 6, "short": 4, "square_length": 0.128,
                          "marker_length": 0.10, "aruco_type": "4X4_50"},
    "lab_charuco_12x8":  {"long": 12, "short": 8, "square_length": 0.16,
                          "marker_length": 0.10, "aruco_type": "4X4_1000"},
}


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


# --------------------------- BOARD ----------------------------------------

def create_board(aruco_type, long, short, square_length, marker_length, custom_ids=None):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    ids = None
    if custom_ids is not None:
        ids = np.asarray(custom_ids, dtype=np.int32).reshape(-1, 1)
    if hasattr(cv2.aruco, "CharucoBoard"):
        if ids is None:
            board = cv2.aruco.CharucoBoard((long, short), square_length,
                                           marker_length, dictionary)
        else:
            board = cv2.aruco.CharucoBoard((long, short), square_length,
                                           marker_length, dictionary, ids)
    else:
        board = cv2.aruco.CharucoBoard_create(long, short, square_length,
                                              marker_length, dictionary)
    corners3d = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") \
                else board.chessboardCorners
    return dictionary, board, np.asarray(corners3d, dtype=np.float32)


def make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = min(params.minMarkerPerimeterRate, 0.01)
    return params


def detect_charuco_in_image(gray, dictionary, board, params, scales):
    """Multi-scale detection, returns (ids, corners2d) or (None, None)."""
    for scale in scales:
        work = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if hasattr(cv2.aruco, "ArucoDetector"):
            det = cv2.aruco.ArucoDetector(dictionary, params)
            mcorners, mids, _ = det.detectMarkers(work)
        else:
            mcorners, mids, _ = cv2.aruco.detectMarkers(work, dictionary,
                                                       parameters=params)
        if mids is None or len(mids) == 0:
            continue
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=mcorners, markerIds=mids, image=work, board=board)
        if not ret or ch_ids is None:
            continue
        ch_ids = ch_ids.reshape(-1).astype(np.int32)
        ch_corners = ch_corners.reshape(-1, 2).astype(np.float32)
        if scale != 1.0:
            ch_corners = ch_corners / scale
        return ch_ids, ch_corners
    return None, None


# ------------------------ DETECTION DRIVER --------------------------------

def detect_all(image_root, cams, ext, dictionary, board, scales,
               step, frames_filter, min_corners, verbose=True):
    """Returns dict cam -> {frame: (ids ndarray, corners2d ndarray)}."""
    params = make_detector_params()
    out = {c: {} for c in cams}
    for cam in cams:
        imgs = sorted(glob(join(image_root, cam, f"*{ext}")))
        if step > 1:
            imgs = imgs[::step]
        if frames_filter is not None:
            imgs = [p for p in imgs
                    if os.path.splitext(os.path.basename(p))[0] in frames_filter]
        n_ok = 0
        for p in imgs:
            frame = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ids, corners = detect_charuco_in_image(
                gray, dictionary, board, params, scales)
            if ids is None or ids.shape[0] < min_corners:
                continue
            out[cam][frame] = (ids, corners)
            n_ok += 1
        if verbose:
            print(f"[detect] {cam}: scanned={len(imgs)}, valid(>={min_corners})={n_ok}")
    return out


# ---------------------- INITIAL POSE ESTIMATION ---------------------------

def pose_to_mat(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def mat_to_rt(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3), T[:3, 3].reshape(3)


def solve_pnp_per_view(detections, corners3d_all, intri, min_corners):
    """Returns pnp[cam][frame] = (T_cam_from_board 4x4, n_pts).
    Drops frames where solvePnP fails or reproj > 8px (initial sanity)."""
    pnp = {c: {} for c in detections}
    for cam in detections:
        K = intri[cam]["K"]
        dist = intri[cam]["dist"]
        for frame, (ids, corners) in detections[cam].items():
            if ids.shape[0] < min_corners:
                continue
            obj = corners3d_all[ids].astype(np.float32)
            img = corners.astype(np.float32)
            try:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                ok = False
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean()
            if err > 8.0:
                continue
            pnp[cam][frame] = (pose_to_mat(rvec, tvec), int(ids.shape[0]))
    return pnp


def choose_ref_cam(pnp, prefer=None):
    counts = {c: len(pnp[c]) for c in pnp}
    if prefer is not None and prefer in counts and counts[prefer] > 0:
        return prefer
    return max(counts, key=lambda c: counts[c])


def average_rotations(Rs):
    """Quaternion average. Rs: list of 3x3."""
    Q = []
    for R in Rs:
        # rotation matrix -> quaternion (w,x,y,z)
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


def init_cam_poses(pnp, ref_cam):
    """For each cam c != ref, T_{c<-ref} = T_{c<-board}^f @ inv(T_{ref<-board}^f),
    averaged across frames where both have PnP.
    Returns dict cam -> T_{c<-world} (4x4) with world = ref camera frame."""
    cam_T = {ref_cam: np.eye(4)}
    ref_frames = set(pnp[ref_cam].keys())
    for cam in pnp:
        if cam == ref_cam:
            continue
        common = ref_frames & set(pnp[cam].keys())
        if not common:
            print(f"[init] WARNING: {cam} shares 0 frames with ref {ref_cam}; "
                  f"skipping (will be added if reachable through chain).")
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
        print(f"[init] {cam}: averaged over {len(Rs)} frames")
    return cam_T


def init_board_poses(pnp, ref_cam, cam_T, used_frames):
    """T_{world<-board}^f for each frame.
    Prefer ref cam; if missing, derive via any cam c that has both T_c known and PnP."""
    board_T = {}
    for f in used_frames:
        if f in pnp[ref_cam]:
            T_r_b = pnp[ref_cam][f][0]            # ref-from-board
            board_T[f] = np.linalg.inv(T_r_b)     # world-from-board (world=ref)
            continue
        # fallback: any other cam
        for cam in pnp:
            if cam == ref_cam or cam not in cam_T:
                continue
            if f not in pnp[cam]:
                continue
            T_c_b = pnp[cam][f][0]
            T_c_w = cam_T[cam]
            T_w_b = np.linalg.inv(T_c_w) @ T_c_b
            board_T[f] = T_w_b
            break
    return board_T


# ---------------------------- BUNDLE ADJUSTMENT ---------------------------

def build_observations(detections, used_cams, used_frames, corners3d_all):
    """List of (cam_idx, frame_idx, corner_id_idx_in_obs, uv)."""
    obs = []
    frame_idx = {f: i for i, f in enumerate(used_frames)}
    for ci, cam in enumerate(used_cams):
        for f in detections[cam]:
            if f not in frame_idx:
                continue
            ids, corners = detections[cam][f]
            for k, kid in enumerate(ids):
                obs.append((ci, frame_idx[f], int(kid), corners[k]))
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


def project_points_vectorized(P3, rvec_c, tvec_c, rvec_b, tvec_b, K, dist):
    """P3 (N,3) board-local. Compose board->world->cam, then project with dist."""
    Rb, _ = cv2.Rodrigues(rvec_b.reshape(3, 1))
    Pw = (Rb @ P3.T).T + tvec_b.reshape(1, 3)
    Rc, _ = cv2.Rodrigues(rvec_c.reshape(3, 1))
    Pc = (Rc @ Pw.T).T + tvec_c.reshape(1, 3)
    # Use cv2.projectPoints for distortion; in pure cam frame so rvec/tvec=0
    proj, _ = cv2.projectPoints(Pc.astype(np.float32),
                                np.zeros(3), np.zeros(3), K, dist)
    return proj.reshape(-1, 2)


def run_ba(observations, used_cams, used_frames, corners3d_all,
           intri, cam_Ts, board_Ts, ref_cam,
           huber_pixels=2.0, max_nfev=120, verbose=2):
    ref_idx = used_cams.index(ref_cam)
    n_cams = len(used_cams)
    n_frames = len(used_frames)
    n_obs = len(observations)
    print(f"[ba] cams={n_cams} (ref={ref_cam}), frames={n_frames}, "
          f"observations={n_obs}")

    Ks = [intri[c]["K"] for c in used_cams]
    Ds = [intri[c]["dist"] for c in used_cams]

    # Group observations by (cam, frame) for batch project
    by_cf = {}
    for oi, (ci, fi, kid, uv) in enumerate(observations):
        by_cf.setdefault((ci, fi), []).append((oi, kid, uv))

    target_uv = np.zeros((n_obs, 2))
    for oi, (_, _, _, uv) in enumerate(observations):
        target_uv[oi] = uv

    def residual(x):
        cam_RT, board_RT = unpack_params(x, n_cams, n_frames, ref_idx)
        proj_all = np.zeros((n_obs, 2))
        for (ci, fi), items in by_cf.items():
            kids = np.array([k for _, k, _ in items], dtype=np.int64)
            P3 = corners3d_all[kids]
            rc, tc = cam_RT[ci]
            rb, tb = board_RT[fi]
            proj = project_points_vectorized(P3, rc, tc, rb, tb, Ks[ci], Ds[ci])
            for k, (oi, _, _) in enumerate(items):
                proj_all[oi] = proj[k]
        r = (proj_all - target_uv).reshape(-1)
        # Huber via scaled residual
        # (least_squares supports loss='huber' directly which is cleaner)
        return r

    # Sparse Jacobian sparsity
    n_params = 6 * (n_cams - 1) + 6 * n_frames
    jac = lil_matrix((2 * n_obs, n_params), dtype=np.uint8)
    for oi, (ci, fi, _, _) in enumerate(observations):
        # cam block
        if ci != ref_idx:
            cam_offset = 6 * (ci if ci < ref_idx else ci - 1)
            for d in range(6):
                jac[2*oi, cam_offset + d] = 1
                jac[2*oi+1, cam_offset + d] = 1
        # board block
        b_offset = 6 * (n_cams - 1) + 6 * fi
        for d in range(6):
            jac[2*oi, b_offset + d] = 1
            jac[2*oi+1, b_offset + d] = 1

    x0 = pack_params(cam_Ts, board_Ts, used_cams, used_frames, ref_idx)

    res0 = residual(x0)
    print(f"[ba] initial RMS = {np.sqrt(np.mean(res0**2)):.3f} px, "
          f"mean |err| = {np.mean(np.abs(res0.reshape(-1, 2)).sum(axis=1)/2):.3f} px")

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
    res1 = residual(result.x)
    print(f"[ba] final RMS = {np.sqrt(np.mean(res1**2)):.3f} px, "
          f"mean |err| = {np.mean(np.abs(res1.reshape(-1, 2)).sum(axis=1)/2):.3f} px")
    return result.x


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


def report_per_cam_error(observations, used_cams, used_frames, corners3d_all,
                        intri, x):
    ref_idx = 0  # not used here; just reuse unpack
    # Actually we need ref_idx; pass via closure of caller. Recompute:
    raise NotImplementedError("call report_per_cam_error_v2 instead")


def report_per_cam_error_v2(observations, used_cams, used_frames, corners3d_all,
                            intri, x, ref_cam):
    ref_idx = used_cams.index(ref_cam)
    n_cams = len(used_cams)
    n_frames = len(used_frames)
    cam_RT, board_RT = unpack_params(x, n_cams, n_frames, ref_idx)

    per = {c: [] for c in used_cams}
    by_cf = {}
    for oi, (ci, fi, kid, uv) in enumerate(observations):
        by_cf.setdefault((ci, fi), []).append((kid, uv))
    for (ci, fi), items in by_cf.items():
        cam = used_cams[ci]
        kids = np.array([k for k, _ in items], dtype=np.int64)
        uvs = np.array([uv for _, uv in items], dtype=np.float32)
        P3 = corners3d_all[kids]
        rc, tc = cam_RT[ci]
        rb, tb = board_RT[fi]
        proj = project_points_vectorized(P3, rc, tc, rb, tb,
                                         intri[cam]["K"], intri[cam]["dist"])
        err = np.linalg.norm(proj - uvs, axis=1)
        per[cam].extend(err.tolist())

    print("\n[report] per-camera reprojection error after BA:")
    print(f"  {'cam':<25} {'n':>6} {'mean':>8} {'median':>8} "
          f"{'p90':>8} {'p95':>8} {'p99':>8}")
    for cam in used_cams:
        e = np.array(per[cam])
        if e.size == 0:
            print(f"  {cam:<25} {0:>6} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue
        print(f"  {cam:<25} {e.size:>6d} {e.mean():>8.3f} "
              f"{np.median(e):>8.3f} {np.percentile(e,90):>8.3f} "
              f"{np.percentile(e,95):>8.3f} {np.percentile(e,99):>8.3f}")


# --------------------------------- MAIN -----------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Multi-frame Charuco extri calibration with bundle adjustment")
    p.add_argument("path", type=str, help="dataset root (must contain images/<cam>/*.jpg)")
    p.add_argument("--image", type=str, default="images")
    p.add_argument("--ext", type=str, default=".jpg")
    p.add_argument("--out", type=str, default=None,
                   help="output dir; default <path>/output_charuco_ba")
    p.add_argument("--intri", type=str, required=True,
                   help="intri.yml from intrinsics calibration")
    p.add_argument("--cams", type=str, default=None,
                   help="csv of cameras to use; default = all under images/")
    p.add_argument("--ref_cam", type=str, default=None,
                   help="reference camera (world origin); default = best coverage")
    # board
    p.add_argument("--template", type=str, default="lab_charuco_12x8",
                   choices=sorted(BOARD_TEMPLATES.keys()))
    p.add_argument("--long", type=int, default=None)
    p.add_argument("--short", type=int, default=None)
    p.add_argument("--square_length", type=float, default=None)
    p.add_argument("--marker_length", type=float, default=None)
    p.add_argument("--aruco_type", type=str, default=None)
    p.add_argument("--id_layout_json", type=str, default=None,
                   help="custom board_order_ids json (reconstructed layout)")
    # detection
    p.add_argument("--step", type=int, default=10,
                   help="subsample frames by this step (per camera)")
    p.add_argument("--frames", type=str, default=None,
                   help="csv of frame names to include (override step)")
    p.add_argument("--scales", type=str, default="1,2",
                   help="multi-scale detection scales")
    p.add_argument("--min_corners", type=int, default=8,
                   help="min charuco corners per (cam, frame) to keep")
    # BA
    p.add_argument("--huber_px", type=float, default=2.0)
    p.add_argument("--max_nfev", type=int, default=120)
    p.add_argument("--max_frames", type=int, default=600,
                   help="cap on board-pose unknowns; uniformly subsample if exceeded")
    args = p.parse_args()

    # Resolve board params
    preset = BOARD_TEMPLATES[args.template]
    long_ = args.long if args.long is not None else preset["long"]
    short_ = args.short if args.short is not None else preset["short"]
    sq = args.square_length if args.square_length is not None else preset["square_length"]
    ml = args.marker_length if args.marker_length is not None else preset["marker_length"]
    at = args.aruco_type if args.aruco_type is not None else preset["aruco_type"]

    custom_ids = None
    if args.id_layout_json is not None:
        layout = json.load(open(args.id_layout_json))
        custom_ids = layout.get("board_order_ids")
        if "aruco_type" in layout and args.aruco_type is None:
            at = str(layout["aruco_type"])
        if "long" in layout:
            long_ = int(layout["long"])
        if "short" in layout:
            short_ = int(layout["short"])
        print(f"[layout] using custom ids from {args.id_layout_json} (n={len(custom_ids)})")

    print(f"[board] long={long_}, short={short_}, square={sq}, marker={ml}, aruco={at}")
    dictionary, board, corners3d_all = create_board(at, long_, short_, sq, ml, custom_ids)
    print(f"[board] {corners3d_all.shape[0]} chessboard corners total")

    # Resolve cameras
    image_root = join(args.path, args.image)
    all_cams = list_cameras(image_root)
    if args.cams is not None:
        wanted = [c.strip() for c in args.cams.split(",") if c.strip()]
        used_cams = [c for c in wanted if c in all_cams]
    else:
        used_cams = all_cams
    print(f"[cams] using {len(used_cams)} cameras: {used_cams}")

    # Intri
    intri = read_intri(args.intri)
    missing = [c for c in used_cams if c not in intri]
    if missing:
        raise RuntimeError(f"intri.yml missing cameras: {missing}")

    # Frames filter
    frames_filter = None
    if args.frames is not None:
        frames_filter = set([x.strip() for x in args.frames.split(",") if x.strip()])

    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    # Detect
    detections = detect_all(image_root, used_cams, args.ext,
                            dictionary, board, scales,
                            args.step, frames_filter, args.min_corners)

    n_total = sum(len(detections[c]) for c in used_cams)
    if n_total == 0:
        raise RuntimeError("No charuco detections at all. Check board params / images.")

    # Per-view PnP
    pnp = solve_pnp_per_view(detections, corners3d_all, intri, args.min_corners)
    for c in used_cams:
        print(f"[pnp] {c}: {len(pnp[c])} frames")

    # Pick reference camera
    ref_cam = choose_ref_cam(pnp, prefer=args.ref_cam)
    print(f"[ref] reference camera = {ref_cam}")

    # Initial camera poses
    cam_Ts = init_cam_poses(pnp, ref_cam)
    missing_init = [c for c in used_cams if c not in cam_Ts]
    if missing_init:
        # Try to chain through any cam that has init pose
        print(f"[init] WARNING: cameras with no shared frame with ref: {missing_init}")
        for cam in missing_init:
            for via in cam_Ts:
                if via == ref_cam:
                    continue
                common = set(pnp[cam].keys()) & set(pnp[via].keys())
                if not common:
                    continue
                Rs, Ts = [], []
                for f in common:
                    T_c_b = pnp[cam][f][0]
                    T_v_b = pnp[via][f][0]
                    T_c_v = T_c_b @ np.linalg.inv(T_v_b)
                    T_c_w = T_c_v @ cam_Ts[via]
                    Rs.append(T_c_w[:3, :3])
                    Ts.append(T_c_w[:3, 3])
                if Rs:
                    R_avg = average_rotations(Rs)
                    T_avg = np.median(np.stack(Ts), axis=0)
                    T_full = np.eye(4)
                    T_full[:3, :3] = R_avg
                    T_full[:3, 3] = T_avg
                    cam_Ts[cam] = T_full
                    print(f"[init] {cam}: chained via {via} (n={len(Rs)})")
                    break
    still_missing = [c for c in used_cams if c not in cam_Ts]
    if still_missing:
        print(f"[init] dropping cameras with no path to ref: {still_missing}")
        used_cams = [c for c in used_cams if c in cam_Ts]

    # Decide used frames: where >= 2 cams have PnP
    frame_cam_count = {}
    for cam in used_cams:
        for f in pnp[cam]:
            frame_cam_count[f] = frame_cam_count.get(f, 0) + 1
    used_frames = sorted([f for f, n in frame_cam_count.items() if n >= 2])
    if len(used_frames) == 0:
        raise RuntimeError("No frame is seen by >= 2 cameras with valid PnP.")
    if len(used_frames) > args.max_frames:
        idx = np.linspace(0, len(used_frames) - 1, args.max_frames, dtype=int)
        used_frames = [used_frames[i] for i in idx]
    print(f"[frames] using {len(used_frames)} frames in BA")

    board_Ts = init_board_poses(pnp, ref_cam, cam_Ts, used_frames)
    # Drop frames whose board pose couldn't be initialized
    used_frames = [f for f in used_frames if f in board_Ts]
    print(f"[frames] {len(used_frames)} frames with valid board init")

    observations = build_observations(detections, used_cams, used_frames, corners3d_all)
    print(f"[obs] total observations = {len(observations)}")

    # BA
    x_opt = run_ba(observations, used_cams, used_frames, corners3d_all,
                   intri, cam_Ts, board_Ts, ref_cam,
                   huber_pixels=args.huber_px, max_nfev=args.max_nfev)

    # Report
    report_per_cam_error_v2(observations, used_cams, used_frames, corners3d_all,
                            intri, x_opt, ref_cam)

    # Write extri.yml
    out_dir = args.out or join(args.path, "output_charuco_ba")
    os.makedirs(out_dir, exist_ok=True)
    extri = assemble_extri(x_opt, used_cams, used_frames, ref_cam)
    extri_path = join(out_dir, "extri.yml")
    write_extri(extri_path, extri, used_cams)
    print(f"\n[write] extri.yml -> {extri_path}")

    # Copy intri.yml alongside (so the dir is self-contained)
    intri_out = join(out_dir, "intri.yml")
    shutil.copy(args.intri, intri_out)
    print(f"[write] intri.yml -> {intri_out}")

    # Save a meta summary
    meta = {
        "ref_cam": ref_cam,
        "cams": used_cams,
        "n_frames_used": len(used_frames),
        "n_observations": len(observations),
        "board": {"long": long_, "short": short_,
                  "square_length": sq, "marker_length": ml,
                  "aruco_type": at},
        "step": args.step,
        "min_corners": args.min_corners,
        "huber_px": args.huber_px,
    }
    with open(join(out_dir, "ba_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] ba_meta.json -> {join(out_dir, 'ba_meta.json')}")


if __name__ == "__main__":
    main()
