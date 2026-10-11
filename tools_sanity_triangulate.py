"""Sanity-check: verify the extri.yml is internally consistent.

For each cam:
  1. Detect 5x3 chessboard.
  2. solvePnP using its known K + dist → T_cam_from_board (R, t).
  3. Multiply by cam's extri (T_cam_from_world from extri.yml) → derive
     T_world_from_board for that cam.

All cams should agree on T_world_from_board (since they observe the same
physical board). Disagreement reveals extri error.

Also reports per-cam reprojection error using a canonical world board pose
(from a low-error cam), handling 180-degree corner-ordering ambiguity by
testing both orderings and picking the better one per cam.
"""

import argparse
import json
import os
import subprocess

import cv2
import numpy as np


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real()))
             for i in range(nn.size())]
    intri = {c: {"K": fs.getNode(f"K_{c}").mat(),
                 "dist": fs.getNode(f"dist_{c}").mat()} for c in names}
    fs.release(); return intri


def read_extri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real()))
             for i in range(nn.size())]
    extri = {c: {"R": fs.getNode(f"Rot_{c}").mat(),
                 "T": fs.getNode(f"T_{c}").mat()} for c in names}
    fs.release(); return extri


def detect(gray, pattern=(5, 3)):
    pat_alt = (pattern[1], pattern[0])
    for pat in (pattern, pat_alt):
        try:
            ok, c = cv2.findChessboardCorners(
                gray, pat,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                      cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_FAST_CHECK)
        except cv2.error: ok = False
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
            except cv2.error: ok = False
            if ok and c is not None:
                return pat, c.reshape(-1, 2).astype(np.float32)
    return None, None


def build_corners3d(pat, grid):
    cols, rows = pat
    pts = np.zeros((cols*rows, 3), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            pts[idx, 0] = c * grid; pts[idx, 1] = r * grid; idx += 1
    return pts


def parse_time(s):
    p = s.split(':'); return int(p[0])*3600+int(p[1])*60+float(p[2])


def extract_frame(video, t_in_v):
    cmd = ['ffmpeg','-y','-ss',f'{t_in_v:.3f}','-i',video,'-vframes','1',
           '-f','image2pipe','-vcodec','mjpeg','pipe:1']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout: return None
    return cv2.imdecode(np.frombuffer(r.stdout, dtype=np.uint8),
                         cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync_meta", required=True)
    ap.add_argument("--intri", required=True)
    ap.add_argument("--extri", required=True)
    ap.add_argument("--t_wall", default="13:02:30")
    ap.add_argument("--grid", type=float, default=0.19)
    args = ap.parse_args()

    intri = read_intri(args.intri); extri = read_extri(args.extri)
    meta = json.load(open(args.sync_meta))
    t_wall = parse_time(args.t_wall)

    # detect + per-cam T_cam_from_board via solvePnP
    detections = {}
    Tcb_per_cam = {}
    print("step 1: detect + per-cam solvePnP")
    for cam, info in meta["cameras"].items():
        t_in_v = t_wall - info["start_sec_since_midnight"]
        if t_in_v < 0 or t_in_v > info["duration_sec"]: continue
        img = extract_frame(info["video"], t_in_v)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pat, c2d = detect(gray, (5, 3))
        if c2d is None:
            print(f"  {cam[:3]}: no detection"); continue
        obj = build_corners3d(pat, args.grid)
        try:
            ok, rv, tv = cv2.solvePnP(obj, c2d, intri[cam]["K"], intri[cam]["dist"],
                                       flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error: ok = False
        if not ok:
            print(f"  {cam[:3]}: solvePnP failed"); continue
        proj, _ = cv2.projectPoints(obj, rv, tv, intri[cam]["K"], intri[cam]["dist"])
        err = float(np.linalg.norm(proj.reshape(-1, 2) - c2d, axis=1).mean())
        Rcb, _ = cv2.Rodrigues(rv)
        Tcb = np.eye(4); Tcb[:3, :3] = Rcb; Tcb[:3, 3] = tv.reshape(3)
        detections[cam] = (pat, c2d)
        Tcb_per_cam[cam] = (Tcb, err)
        print(f"  {cam[:3]}: pat={pat}, single-cam reproj={err:.2f}px")

    # derive T_world_from_board from each cam separately, compare
    print("\nstep 2: T_world_from_board derived from each cam")
    print(f"  {'cam':<10}{'tx':>9}{'ty':>9}{'tz':>9}{'rmag(deg)':>12}")
    print("  " + "-" * 50)
    Twb_per_cam = {}
    for cam, (Tcb, _) in Tcb_per_cam.items():
        R_cw = extri[cam]["R"]; t_cw = extri[cam]["T"].reshape(3)
        # cam → world is the INVERSE of world → cam (where extri = world → cam)
        R_wc = R_cw.T; t_wc = -R_wc @ t_cw
        T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = t_wc
        Twb = T_wc @ Tcb
        Twb_per_cam[cam] = Twb
        r_wb, _ = cv2.Rodrigues(Twb[:3, :3])
        rmag = float(np.linalg.norm(r_wb))
        print(f"  {cam[:3]:<10}{Twb[0,3]:>9.4f}{Twb[1,3]:>9.4f}{Twb[2,3]:>9.4f}"
              f"{np.degrees(rmag):>12.2f}")

    # consistency: how much do the cams' Twb estimates disagree?
    ts = np.array([T[:3, 3] for T in Twb_per_cam.values()])
    print(f"\n  translation spread (m):  std={ts.std(0)} -- ||std||={np.linalg.norm(ts.std(0)):.4f}")
    print(f"  translation range (m):   {ts.max(0)-ts.min(0)} -- max range={(ts.max(0)-ts.min(0)).max():.4f}")

    # pick the cam with lowest single-cam reproj err as canonical board pose
    canonical_cam = min(Tcb_per_cam, key=lambda c: Tcb_per_cam[c][1])
    print(f"\nstep 3: use {canonical_cam[:3]} as canonical board pose source"
          f" (single-cam reproj={Tcb_per_cam[canonical_cam][1]:.2f}px)")
    Twb_ref = Twb_per_cam[canonical_cam]
    pat_ref = detections[canonical_cam][0]
    obj_local = build_corners3d(pat_ref, args.grid)
    Rwb, twb = Twb_ref[:3, :3], Twb_ref[:3, 3]
    world_corners = (Rwb @ obj_local.T).T + twb.reshape(1, 3)

    # for each cam, project world_corners, allow both orderings
    print(f"\nstep 4: per-cam reproj error using canonical world board")
    print(f"  {'cam':<10}{'pat':>8}{'n':>5}{'mean':>9}{'median':>9}{'max':>9}{'ordering':>12}")
    print("  " + "-" * 65)
    all_per_cam = {}
    for cam in meta["cameras"]:
        if cam not in detections: continue
        pat, c2d = detections[cam]
        if pat != pat_ref:
            # would need to handle pattern transpose; skip
            print(f"  {cam[:3]:<10} pattern mismatch ({pat} vs {pat_ref}); skipping")
            continue
        K = intri[cam]["K"]; D = intri[cam]["dist"]
        R = extri[cam]["R"]; t = extri[cam]["T"].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(world_corners.astype(np.float32), rvec, t, K, D)
        proj = proj.reshape(-1, 2)
        err_fwd = np.linalg.norm(proj - c2d, axis=1)
        err_rev = np.linalg.norm(proj - c2d[::-1], axis=1)
        if err_fwd.mean() <= err_rev.mean():
            err = err_fwd; tag = "fwd"
        else:
            err = err_rev; tag = "rev"
        all_per_cam[cam] = err
        print(f"  {cam[:3]:<10}{str(pat):>8}{len(err):>5}{err.mean():>9.3f}"
              f"{np.median(err):>9.3f}{err.max():>9.3f}{tag:>12}")

    print(f"\nstep 5: triangulate 15 corners from all cams (handling ordering)")
    # build per-cam undistorted corners with correct ordering
    P = {}; u2d = {}
    for cam, err in all_per_cam.items():
        K = intri[cam]["K"]; D = intri[cam]["dist"]
        R = extri[cam]["R"]; t = extri[cam]["T"].reshape(3, 1)
        proj, _ = cv2.projectPoints(world_corners.astype(np.float32),
                                     cv2.Rodrigues(R)[0], t, K, D)
        proj = proj.reshape(-1, 2)
        c2d = detections[cam][1]
        if np.linalg.norm(proj - c2d, axis=1).mean() <= \
           np.linalg.norm(proj - c2d[::-1], axis=1).mean():
            c2d_aligned = c2d
        else:
            c2d_aligned = c2d[::-1]
        P[cam] = K @ np.hstack([R, t])
        u = cv2.undistortPoints(c2d_aligned.reshape(-1,1,2), K, D, P=K)
        u2d[cam] = u.reshape(-1, 2)
    # multi-view DLT per corner
    cams = list(P.keys())
    n_corners = len(world_corners)
    tri = np.zeros((n_corners, 3))
    for i in range(n_corners):
        A = []
        for cam in cams:
            Pc = P[cam]; x, y = u2d[cam][i]
            A.append(x*Pc[2] - Pc[0])
            A.append(y*Pc[2] - Pc[1])
        A = np.asarray(A)
        _, _, Vt = np.linalg.svd(A); Xh = Vt[-1]
        tri[i] = Xh[:3] / Xh[3]

    # check grid spacing on triangulated points
    cols, rows = pat_ref
    rd = [np.linalg.norm(tri[r*cols+c+1] - tri[r*cols+c])
          for r in range(rows) for c in range(cols - 1)]
    cd = [np.linalg.norm(tri[(r+1)*cols+c] - tri[r*cols+c])
          for r in range(rows - 1) for c in range(cols)]
    rd, cd = np.array(rd), np.array(cd)

    # planarity check
    cen = tri.mean(0)
    _,_,Vt = np.linalg.svd(tri - cen)
    plane_resid = np.abs((tri - cen) @ Vt[-1])

    print(f"\n[grid] row spacing : mean={rd.mean()*1000:.1f} mm, "
          f"std={rd.std()*1000:.2f} mm, err vs 190mm = {(rd.mean()-args.grid)*1000:+.1f} mm")
    print(f"[grid] col spacing : mean={cd.mean()*1000:.1f} mm, "
          f"std={cd.std()*1000:.2f} mm, err vs 190mm = {(cd.mean()-args.grid)*1000:+.1f} mm")
    print(f"[planar] residual  : mean={plane_resid.mean()*1000:.2f} mm, "
          f"max={plane_resid.max()*1000:.2f} mm")

    print("\nverdict:")
    if rd.mean() > 0 and abs(rd.mean()-args.grid)*1000 < 5 \
       and abs(cd.mean()-args.grid)*1000 < 5 \
       and plane_resid.max()*1000 < 5:
        print("  ✓ extrinsics CONSISTENT — grid recovered to within 5mm, board is planar")
    else:
        print("  ✗ extrinsics need work — grid or planarity off")


if __name__ == "__main__":
    main()
