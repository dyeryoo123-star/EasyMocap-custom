import argparse
import json
import os
from glob import glob
from os.path import basename, join, splitext

import cv2
import numpy as np


MODES = ["identity", "flip_lr", "flip_ud", "flip_both"]


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    node = fs.getNode("names")
    cams = []
    for i in range(node.size()):
        s = node.at(i).string()
        if s == "":
            s = str(int(node.at(i).real()))
        cams.append(s)
    intri = {}
    for cam in cams:
        K = fs.getNode(f"K_{cam}").mat()
        dist = fs.getNode(f"dist_{cam}").mat()
        if K is None or K.size == 0:
            continue
        if dist is None:
            dist = np.zeros((1, 5), dtype=np.float64)
        intri[cam] = (K, dist)
    fs.release()
    return intri


def remap(k2d, pattern, mode):
    w, h = pattern
    arr = k2d.reshape(h, w, 3)
    if mode == "identity":
        arr2 = arr
    elif mode == "flip_lr":
        arr2 = arr[:, ::-1, :]
    elif mode == "flip_ud":
        arr2 = arr[::-1, :, :]
    elif mode == "flip_both":
        arr2 = arr[::-1, ::-1, :]
    else:
        raise ValueError(mode)
    return arr2.reshape(-1, 3)


def parse_csv(s):
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def solve_err(k3d, k2d, K, dist):
    valid = k2d[:, 2] > 0.01
    if int(valid.sum()) < 4:
        return np.nan
    obj = np.ascontiguousarray(k3d[valid].astype(np.float32))
    img = np.ascontiguousarray(k2d[valid, :2].astype(np.float32))
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img, axis=1).mean()
    return float(err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root, e.g. extri_selected_5f_7500_9500")
    parser.add_argument("--intri", type=str, required=True, help="intri.yml path")
    parser.add_argument("--chessboard", type=str, default="chessboard")
    parser.add_argument("--ext", type=str, default=".json")
    parser.add_argument("--use_backup", action="store_true", help="read *.json.bak")
    parser.add_argument("--cams", type=str, default=None, help="camera list csv")
    parser.add_argument("--frames", type=str, default=None, help="frame list csv, like 000000,000001")
    args = parser.parse_args()

    intri = read_intri(args.intri)
    chess_root = join(args.path, args.chessboard)
    cams = sorted([d for d in os.listdir(chess_root) if os.path.isdir(join(chess_root, d))])
    cam_filter = set(parse_csv(args.cams))
    if len(cam_filter) > 0:
        cams = [c for c in cams if c in cam_filter]
    frames_filter = set(parse_csv(args.frames))

    print("[info] cameras:", cams)
    print("[info] use_backup:", args.use_backup)

    # camera -> mode -> frame -> err
    result = {}
    all_frames = set()
    for cam in cams:
        if cam not in intri:
            print(f"[skip] {cam}: missing intrinsics")
            continue
        K, dist = intri[cam]
        suffix = ".json.bak" if args.use_backup else ".json"
        files = sorted(glob(join(chess_root, cam, f"*{suffix}")))
        if len(files) == 0:
            print(f"[skip] {cam}: no {suffix} files")
            continue
        result[cam] = {m: {} for m in MODES}
        for fp in files:
            frame = splitext(basename(fp))[0]
            if frame.endswith(".json"):
                # basename for .json.bak gives "000000.json"
                frame = frame[:-5]
            if len(frames_filter) > 0 and frame not in frames_filter:
                continue
            data = json.load(open(fp, "r"))
            k3d = np.asarray(data["keypoints3d"], dtype=np.float32)
            k2d = np.asarray(data["keypoints2d"], dtype=np.float32)
            if "pattern" in data:
                pattern = (int(data["pattern"][0]), int(data["pattern"][1]))
            else:
                # fallback: infer 5x3 from 15 etc. only for known small board
                n = int(k2d.shape[0])
                if n == 15:
                    pattern = (5, 3)
                else:
                    raise ValueError(f"{cam}/{frame}: missing pattern and cannot infer from n={n}")
            if k2d.shape[0] != pattern[0] * pattern[1]:
                print(f"[warn] {cam}/{frame}: mismatch keypoints and pattern")
                continue
            all_frames.add(frame)
            for mode in MODES:
                kk = remap(k2d, pattern, mode)
                err = solve_err(k3d, kk, K, dist)
                result[cam][mode][frame] = err

    # per-camera summary
    cam_best = {}
    for cam in sorted(result.keys()):
        print(f"\n[{cam}]")
        scored = []
        for mode in MODES:
            vals = np.array(
                [v for v in result[cam][mode].values() if np.isfinite(v)],
                dtype=np.float32,
            )
            if vals.size == 0:
                print(f"  {mode:10s} no valid frame")
                continue
            med = float(np.median(vals))
            mean = float(np.mean(vals))
            mx = float(np.max(vals))
            scored.append((med, mean, mode))
            print(f"  {mode:10s} median={med:.3f} mean={mean:.3f} max={mx:.3f} n={vals.size}")
        if len(scored) > 0:
            scored.sort(key=lambda x: (x[0], x[1]))
            cam_best[cam] = scored[0][2]
            print(f"  -> suggest mode: {cam_best[cam]}")

    # frame summary with suggested camera mode
    frames = sorted(all_frames)
    if len(frames) == 0:
        print("\n[summary] no valid frames")
        return
    print("\n[frame ranking using suggested mode]")
    frame_scores = []
    for fr in frames:
        vals = []
        for cam in sorted(cam_best.keys()):
            mode = cam_best[cam]
            e = result[cam][mode].get(fr, np.nan)
            if np.isfinite(e):
                vals.append(float(e))
        if len(vals) == 0:
            continue
        score_mean = float(np.mean(vals))
        score_max = float(np.max(vals))
        frame_scores.append((score_mean, score_max, fr, len(vals)))
    frame_scores.sort(key=lambda x: (x[0], x[1], -x[3]))
    for mean, mx, fr, n in frame_scores:
        print(f"  frame={fr} mean={mean:.3f} max={mx:.3f} cams={n}")

    if len(cam_best) > 0:
        apply_map = ",".join([f"{k}:{v}" for k, v in sorted(cam_best.items())])
        print("\n[suggest apply_map]")
        print(apply_map)


if __name__ == "__main__":
    main()
