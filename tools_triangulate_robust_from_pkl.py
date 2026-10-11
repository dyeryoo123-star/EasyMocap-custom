"""Robustly triangulate BODY25 3D keypoints from saved 2D PKL detections.

This is a lightweight alternative to re-running the full EasyMocap detector:
it reads existing `keypoints2d/<cam>/<frame>.pkl`, uses a selected camera
subset, rejects high-reprojection outlier views per joint, and writes
EasyMocap-style `keypoints3d/*.json` files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from easymocap.mytools.camera_utils import read_camera
from myeasymocap.backbone.vitpose.vit_plain import coco25tobody25


BODY25_SKELETON = [
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16), (15, 17), (16, 18),
    (11, 24), (11, 22), (22, 23), (14, 21), (14, 19), (19, 20),
]


def parse_cams(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_offsets(text: str, cams: List[str]) -> Dict[str, int]:
    offsets = {cam: 0 for cam in cams}
    if not text:
        return offsets
    if os.path.exists(text):
        data = json.load(open(text))
        raw = data.get("lags", data)
        for cam in cams:
            offsets[cam] = int(raw.get(cam, raw.get(cam[:3], 0)))
        return offsets
    for item in text.split(","):
        if not item.strip():
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        for cam in cams:
            if cam == key or cam[:3] == key:
                offsets[cam] = int(val)
    return offsets


def load_pkl(path: Path, n_joints: int = 25, input_order: str = "coco25") -> np.ndarray:
    if not path.exists():
        return np.zeros((n_joints, 3), dtype=np.float32)
    with open(path, "rb") as f:
        data = pickle.load(f)
    kpts = np.asarray(data.get("keypoints", []), dtype=np.float32)
    if kpts.ndim == 3:
        kpts = kpts[0]
    out = np.zeros((n_joints, 3), dtype=np.float32)
    if kpts.size:
        n = min(n_joints, kpts.shape[0])
        out[:n] = kpts[:n]
    if input_order == "coco25":
        # ViTPose's 25-keypoint cache is coco_25; SMPL fitting expects OpenPose BODY25.
        out = coco25tobody25(out[None])[0].astype(np.float32)
    return out


def undistort_kpts(kpts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = kpts.copy()
    valid = out[:, 2] > 0
    if valid.any():
        pts = out[valid, :2].reshape(-1, 1, 2).astype(np.float32)
        und = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
        out[valid, :2] = und
    return out


def triangulate_dlt(kpts: np.ndarray, Ps: np.ndarray) -> np.ndarray:
    """Weighted DLT for one joint across V views. kpts shape: (V,3)."""
    rows = []
    for kp, P in zip(kpts, Ps):
        conf = float(kp[2])
        if conf <= 0:
            continue
        x, y = float(kp[0]), float(kp[1])
        rows.append(conf * (x * P[2] - P[0]))
        rows.append(conf * (y * P[2] - P[1]))
    if len(rows) < 4:
        return np.zeros(4, dtype=np.float32)
    A = np.stack(rows, axis=0)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return np.zeros(4, dtype=np.float32)
    X = vh[-1]
    if abs(X[3]) < 1e-9:
        return np.zeros(4, dtype=np.float32)
    X = X / X[3]
    return X.astype(np.float32)


def project_points(X: np.ndarray, cam: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
    rvec, _ = cv2.Rodrigues(cam["R"])
    T = cam["T"].reshape(3, 1)
    pts, _ = cv2.projectPoints(X[:3].reshape(1, 3).astype(np.float32),
                               rvec, T, cam["K"], cam["dist"])
    Xcam = cam["R"] @ X[:3].reshape(3, 1) + T
    return pts.reshape(2), float(Xcam[2, 0])


def robust_joint(kpts_raw: np.ndarray,
                 kpts_und: np.ndarray,
                 cams: List[str],
                 cameras: Dict[str, Dict[str, np.ndarray]],
                 Ps: np.ndarray,
                 conf_min: float,
                 inlier_px: float,
                 min_inliers: int) -> Tuple[np.ndarray, Dict]:
    valid_idx = [i for i in range(len(cams)) if kpts_raw[i, 2] >= conf_min]
    if len(valid_idx) < 2:
        return np.zeros(4, dtype=np.float32), {"inliers": [], "median_px": None}

    best = None
    for pair in itertools.combinations(valid_idx, 2):
        X = triangulate_dlt(kpts_und[list(pair)], Ps[list(pair)])
        if X[3] == 0:
            continue
        errs = []
        inliers = []
        for i in valid_idx:
            uv, depth = project_points(X, cameras[cams[i]])
            if depth <= 0:
                continue
            err = float(np.linalg.norm(uv - kpts_raw[i, :2]))
            errs.append((i, err))
            if err <= inlier_px:
                inliers.append(i)
        if not errs:
            continue
        arr = np.asarray([e for _, e in errs], dtype=np.float32)
        score = (len(inliers), -float(np.median(arr)), -float(np.percentile(arr, 90)))
        if best is None or score > best[0]:
            best = (score, X, inliers, errs)

    if best is None:
        return np.zeros(4, dtype=np.float32), {"inliers": [], "median_px": None}

    _, X_seed, inliers, errs = best
    if len(inliers) < min_inliers:
        # Keep a 2-view point only when the pair is very self-consistent.
        if len(inliers) < 2:
            return np.zeros(4, dtype=np.float32), {
                "inliers": [cams[i] for i in inliers],
                "median_px": float(np.median([e for _, e in errs])),
            }
        if float(np.median([e for i, e in errs if i in inliers])) > inlier_px * 0.5:
            return np.zeros(4, dtype=np.float32), {
                "inliers": [cams[i] for i in inliers],
                "median_px": float(np.median([e for _, e in errs])),
            }

    X = triangulate_dlt(kpts_und[inliers], Ps[inliers])
    if X[3] == 0:
        X = X_seed

    final_errs = []
    final_inliers = []
    for i in valid_idx:
        uv, depth = project_points(X, cameras[cams[i]])
        if depth <= 0:
            continue
        err = float(np.linalg.norm(uv - kpts_raw[i, :2]))
        if err <= inlier_px:
            final_inliers.append(i)
            final_errs.append(err)

    if len(final_inliers) < 2:
        return np.zeros(4, dtype=np.float32), {"inliers": [], "median_px": None}

    conf = float(np.mean(kpts_raw[final_inliers, 2]))
    Xout = np.zeros(4, dtype=np.float32)
    Xout[:3] = X[:3]
    Xout[3] = conf
    return Xout, {
        "inliers": [cams[i] for i in final_inliers],
        "median_px": float(np.median(final_errs)) if final_errs else None,
        "p90_px": float(np.percentile(final_errs, 90)) if final_errs else None,
    }


def draw_preview(frame_id: str,
                 data_root: Path,
                 kp_root: Path,
                 out_path: Path,
                 cams: List[str],
                 cameras: Dict[str, Dict[str, np.ndarray]],
                 kp3d: np.ndarray,
                 input_order: str,
                 tile_w: int = 640,
                 tile_h: int = 360) -> None:
    tiles = []
    for cam in cams:
        img = cv2.imread(str(data_root / "images" / cam / f"{frame_id}.jpg"))
        if img is None:
            img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        valid = kp3d[:, 3] > 0
        proj = np.full((kp3d.shape[0], 2), np.nan, dtype=np.float32)
        if valid.any():
            rvec, _ = cv2.Rodrigues(cameras[cam]["R"])
            pts, _ = cv2.projectPoints(kp3d[valid, :3].astype(np.float32),
                                       rvec, cameras[cam]["T"].reshape(3, 1),
                                       cameras[cam]["K"], cameras[cam]["dist"])
            proj[valid] = pts.reshape(-1, 2)
        for a, b in BODY25_SKELETON:
            if a < len(proj) and b < len(proj) and valid[a] and valid[b]:
                p1 = tuple(np.round(proj[a]).astype(int))
                p2 = tuple(np.round(proj[b]).astype(int))
                if all(-2000 < v < 5000 for v in p1 + p2):
                    cv2.line(img, p1, p2, (0, 255, 0), 4)
        for j, (x, y) in enumerate(proj):
            if valid[j] and np.isfinite(x) and -2000 < x < 5000 and -2000 < y < 5000:
                cv2.circle(img, (int(round(x)), int(round(y))), 6, (0, 255, 255), -1)
        # Red circles show the original 2D detections used as triangulation input.
        k2d_path = kp_root / cam / f"{frame_id}.pkl"
        k2d = load_pkl(k2d_path, input_order=input_order)
        for x, y, conf in k2d:
            if conf > 0.35:
                cv2.circle(img, (int(round(float(x))), int(round(float(y)))),
                           5, (0, 0, 255), 2)
        img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(img, cam[:3], (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 0), 2)
        tiles.append(img)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows)
    cv2.putText(sheet, f"frame {frame_id} | yellow/green=3D reproj, red=2D",
                (15, sheet.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--kp_root", required=True)
    ap.add_argument("--intri", required=True)
    ap.add_argument("--extri", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--range", default="0,1249")
    ap.add_argument("--offsets", default="")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--inlier_px", type=float, default=120.0)
    ap.add_argument("--min_inliers", type=int, default=3)
    ap.add_argument("--preview_frames", default="0,250,500,750,1000,1249")
    ap.add_argument("--input_order", choices=["coco25", "body25"], default="coco25",
                    help="2D pkl keypoint order. ViTPose caches are usually coco25; "
                         "EasyMocap/SMPL fitting expects BODY25.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    kp_root = Path(args.kp_root)
    out_root = Path(args.out)
    out_k3d = out_root / "keypoints3d"
    out_k3d.mkdir(parents=True, exist_ok=True)

    cams = parse_cams(args.cams)
    offsets = parse_offsets(args.offsets, cams)
    cameras_all = read_camera(args.intri, args.extri)
    cameras_all.pop("basenames", None)
    cameras = {cam: cameras_all[cam] for cam in cams}
    Ps = np.stack([cameras[cam]["P"] for cam in cams], axis=0)

    lo, hi = [int(x) for x in args.range.split(",")]
    preview_frames = {int(x) for x in args.preview_frames.split(",") if x.strip()}

    summary_rows = []
    print(f"[triangulate] cams={cams}")
    print(f"[triangulate] offsets={offsets}")
    print(f"[triangulate] frames={lo}..{hi}")
    for frame in range(lo, hi + 1):
        kpts_raw = []
        kpts_und = []
        for cam in cams:
            src_frame = frame + offsets.get(cam, 0)
            p = kp_root / cam / f"{src_frame:06d}.pkl"
            raw = load_pkl(p, input_order=args.input_order)
            raw[raw[:, 2] < args.conf, 2] = 0
            und = undistort_kpts(raw, cameras[cam]["K"], cameras[cam]["dist"])
            kpts_raw.append(raw)
            kpts_und.append(und)
        kpts_raw_arr = np.stack(kpts_raw, axis=0)
        kpts_und_arr = np.stack(kpts_und, axis=0)

        kp3d = np.zeros((25, 4), dtype=np.float32)
        joint_reports = []
        for j in range(25):
            X, rep = robust_joint(kpts_raw_arr[:, j], kpts_und_arr[:, j],
                                  cams, cameras, Ps, args.conf,
                                  args.inlier_px, args.min_inliers)
            kp3d[j] = X
            joint_reports.append(rep)

        valid = kp3d[:, 3] > 0
        medians = [r["median_px"] for r in joint_reports if r.get("median_px") is not None]
        n_inliers = [len(r.get("inliers", [])) for r in joint_reports if r.get("inliers")]
        out_data = [{
            "id": 0,
            "keypoints3d": kp3d.tolist(),
            "meta": {
                "cams": cams,
                "offsets": offsets,
                "joint_inliers": [r.get("inliers", []) for r in joint_reports],
            },
        }]
        with open(out_k3d / f"{frame:06d}.json", "w") as f:
            json.dump(out_data, f)

        summary_rows.append({
            "frame": frame,
            "valid_joints": int(valid.sum()),
            "median_reproj_px": float(np.median(medians)) if medians else "",
            "p90_joint_median_reproj_px": float(np.percentile(medians, 90)) if medians else "",
            "mean_inlier_views": float(np.mean(n_inliers)) if n_inliers else "",
        })
        if frame in preview_frames:
            draw_preview(f"{frame:06d}", data_root, kp_root, out_root / "preview" / f"reproj_{frame:06d}.jpg",
                         cams, cameras, kp3d, args.input_order)
        if (frame - lo) % 100 == 0:
            print(f"  frame {frame:06d}: valid={int(valid.sum())}/25")

    with open(out_root / "triangulation_quality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(out_root / "run_info.json", "w") as f:
        json.dump({
            "data_root": str(data_root),
            "kp_root": str(kp_root),
            "intri": args.intri,
            "extri": args.extri,
            "cams": cams,
            "offsets": offsets,
            "conf": args.conf,
            "inlier_px": args.inlier_px,
            "min_inliers": args.min_inliers,
            "range": [lo, hi],
            "input_order": args.input_order,
        }, f, indent=2)
    print(f"[wrote] {out_k3d}")
    print(f"[wrote] {out_root / 'triangulation_quality.csv'}")
    print(f"[wrote] {out_root / 'preview'}")


if __name__ == "__main__":
    main()
