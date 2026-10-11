"""Rebase the extrinsic calibration to put the chessboard at the world
origin instead of D01.

Inputs: existing intri.yml + extri.yml (currently D01-as-origin) + a frame
of images where the static 6×4 chessboard is visible to multiple cameras.

Steps:
  1. Detect 5×3 internal corners of the 6×4 board in each cam's image.
  2. Run cv2.solvePnP per cam (using its K + dist + the known 0.19 m board
     geometry) → gives T_cam_from_chess.
  3. For each cam, that T_cam_from_chess IS the new extrinsic relative to
     the chessboard-frame world.  (Mathematically:
       P_cam = T_CW_old @ P_W_old = T_CW_old @ T_WS @ P_S = T_CS @ P_S.)
  4. Pick the cam with the smallest single-frame reprojection error as the
     "reference" so the chessboard-world frame is consistent (other cams
     are propagated via T_CS_cam = T_CW_cam @ T_WS_ref).

Outputs:
  • extri_chess.yml: rebased extrinsics
  • chess_layout_report.json: each cam's position in the new frame,
    so the user can see where they actually sit physically.
"""

import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np


# ---------- IO --------------------------------------------------------------

def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real()))
             for i in range(nn.size())]
    intri = {c: {"K": fs.getNode(f"K_{c}").mat(),
                 "dist": fs.getNode(f"dist_{c}").mat()} for c in names}
    fs.release()
    return intri, names


def read_extri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real()))
             for i in range(nn.size())]
    extri = {}
    for c in names:
        rvec = fs.getNode(f"R_{c}").mat()
        Rot  = fs.getNode(f"Rot_{c}").mat()
        T    = fs.getNode(f"T_{c}").mat()
        extri[c] = {"rvec": rvec, "R": Rot, "T": T}
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
        for c in names_order:
            f.write(f'  - "{c}"\n')
        for c in names_order:
            write_matrix(f, f"R_{c}",   extri[c]["rvec"].reshape(3, 1))
            write_matrix(f, f"Rot_{c}", extri[c]["R"])
            write_matrix(f, f"T_{c}",   extri[c]["T"].reshape(3, 1))


# ---------- chessboard helpers ----------------------------------------------

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


def detect_chessboard(gray, pattern=(5, 3)):
    pat_alt = (pattern[1], pattern[0])
    for pat in (pattern, pat_alt):
        try:
            ok, c = cv2.findChessboardCornersSB(
                gray, pat,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY)
        except cv2.error:
            ok = False
        if ok and c is not None:
            return pat, c.reshape(-1, 2).astype(np.float32)
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
    return None, None


# ---------- main ------------------------------------------------------------

def pose_to_mat(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intri",    required=True)
    ap.add_argument("--extri_in", required=True)
    ap.add_argument("--images_dir", required=True,
                    help="root dir with per-cam subfolders of static board")
    ap.add_argument("--frame", default="000020",
                    help="frame filename stem of the static-board image")
    ap.add_argument("--pattern", default="5x3")
    ap.add_argument("--grid", type=float, default=0.19)
    ap.add_argument("--extri_out", required=True)
    ap.add_argument("--report",    required=True)
    args = ap.parse_args()

    pat = tuple(int(x) for x in args.pattern.split("x"))
    intri, _ = read_intri(args.intri)
    extri_old, names_order = read_extri(args.extri_in)

    # Step 1: per-cam solvePnP on the static chessboard
    obj = build_corners3d(pat, args.grid)
    per_cam_pose = {}    # cam -> (T_cam_from_chess, single-cam reproj err)
    for cam in names_order:
        img_p = os.path.join(args.images_dir, cam, f"{args.frame}.jpg")
        if not os.path.exists(img_p):
            print(f"  {cam}: no image at {img_p}")
            continue
        im = cv2.imread(img_p)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        det_pat, corners = detect_chessboard(gray, pat)
        if corners is None:
            print(f"  {cam}: chessboard not detected")
            continue
        if det_pat != pat:
            # swap object to match detected orientation
            obj_use = build_corners3d(det_pat, args.grid)
        else:
            obj_use = obj
        K = intri[cam]["K"]; D = intri[cam]["dist"]
        ok, rvec, tvec = cv2.solvePnP(obj_use, corners, K, D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print(f"  {cam}: solvePnP failed")
            continue
        proj, _ = cv2.projectPoints(obj_use, rvec, tvec, K, D)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - corners,
                                     axis=1).mean())
        T_CS_pnp = pose_to_mat(rvec, tvec)
        per_cam_pose[cam] = (T_CS_pnp, err, det_pat)
        print(f"  {cam}: detected pat={det_pat}, single-cam reproj={err:.2f}px")

    if len(per_cam_pose) == 0:
        raise RuntimeError("No cameras detected the board — aborting.")

    # Step 2: pick reference cam (lowest single-cam reproj error)
    ref_cam = min(per_cam_pose, key=lambda c: per_cam_pose[c][1])
    print(f"\n[ref] using {ref_cam} (err={per_cam_pose[ref_cam][1]:.2f}px) "
          "as the consistency anchor")

    # Step 3: define T_WS (world_old → chess) from ref cam
    T_CS_ref = per_cam_pose[ref_cam][0]
    # T_CW_ref from extri_old
    R_CW_ref = extri_old[ref_cam]["R"]
    t_CW_ref = extri_old[ref_cam]["T"].reshape(3)
    T_CW_ref = np.eye(4); T_CW_ref[:3, :3] = R_CW_ref; T_CW_ref[:3, 3] = t_CW_ref
    T_WC_ref = np.linalg.inv(T_CW_ref)
    T_WS = T_WC_ref @ T_CS_ref       # chess pose in old world

    # Step 4: rebase each cam to chess world
    extri_new = {}
    cam_positions_new = {}
    for cam in names_order:
        R_CW = extri_old[cam]["R"]
        t_CW = extri_old[cam]["T"].reshape(3)
        T_CW = np.eye(4); T_CW[:3, :3] = R_CW; T_CW[:3, 3] = t_CW
        T_CS_new = T_CW @ T_WS
        R_new = T_CS_new[:3, :3]
        t_new = T_CS_new[:3, 3]
        rvec_new, _ = cv2.Rodrigues(R_new)
        extri_new[cam] = {"rvec": rvec_new, "R": R_new, "T": t_new}
        # camera center in new world
        cam_pos = -R_new.T @ t_new
        cam_positions_new[cam] = cam_pos.tolist()

    # Step 5: write output
    write_extri(args.extri_out, extri_new, names_order)
    print(f"\nwrote {args.extri_out}")

    # Step 6: report
    print(f"\nCamera positions in NEW chess-frame (chess at origin, "
          "X = board col axis, Y = board row axis, Z = board normal):")
    print(f'  {"cam":<6}{"X (m)":>10}{"Y (m)":>10}{"Z (m)":>10}{"|pos|":>10}')
    print("  " + "-" * 50)
    ys = []
    for cam in names_order:
        p = cam_positions_new[cam]
        d = float(np.linalg.norm(p))
        ys.append(p[1])
        print(f"  {cam[:3]:<6}{p[0]:>10.2f}{p[1]:>10.2f}{p[2]:>10.2f}{d:>10.2f}")
    ys = np.array(ys)
    print(f"\nY-axis spread (= height variation if board was held vertical): "
          f"min={ys.min():.2f}m max={ys.max():.2f}m range={ys.ptp():.2f}m")

    report = {
        "ref_cam_for_chess_frame": ref_cam,
        "ref_cam_reproj_err_px": per_cam_pose[ref_cam][1],
        "pattern": list(pat),
        "grid_m": args.grid,
        "cameras_in_new_frame": cam_positions_new,
        "axis_meaning": {
            "X": "along the board's column axis (5 corners side)",
            "Y": "along the board's row axis (3 corners side)",
            "Z": "along the board's outward normal",
        },
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote report {args.report}")


if __name__ == "__main__":
    main()
