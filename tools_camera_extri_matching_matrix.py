"""Build a camera/extrinsic matching matrix from multi-view 2D keypoints.

This diagnostic checks whether each image stream appears to match its named
camera model. It triangulates 3D points from trusted reference cameras, then
projects those 3D points into every image stream using every candidate camera
model. If labels are correct, the matrix diagonal should usually have the
lowest reprojection error.
"""

import argparse
import csv
import json
import os
import pickle
from glob import glob
from os.path import basename, join, splitext

import numpy as np

from easymocap.mytools.camera_utils import read_camera


def parse_csv(value):
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_joints(value):
    vals = parse_csv(value)
    if vals is None:
        return None
    return sorted(set(int(v) for v in vals))


def parse_offsets(value, cams):
    offsets = {cam: 0 for cam in cams}
    if not value:
        return offsets
    if os.path.exists(value):
        with open(value) as f:
            data = json.load(f)
        offsets.update({k: int(v) for k, v in data.get("lags", {}).items()})
        return offsets
    for item in value.split(","):
        if not item.strip():
            continue
        cam, val = item.split(":")
        offsets[cam.strip()] = int(val)
    return offsets


def load_pkl_2d(path, n_joints=25, conf_min=0.0):
    if not os.path.exists(path):
        return np.zeros((n_joints, 3), dtype=np.float32)
    with open(path, "rb") as f:
        data = pickle.load(f)
    k = np.asarray(data.get("keypoints", []), dtype=np.float32)
    if k.ndim == 3:
        k = k[0]
    out = np.zeros((n_joints, 3), dtype=np.float32)
    if k.size == 0:
        return out
    m = min(n_joints, k.shape[0])
    out[:m] = k[:m]
    out[out[:, 2] < conf_min] = 0
    return out


def load_all_keypoints(kp_root, cams, conf_min=0.0):
    annots = {}
    for cam in cams:
        files = sorted(glob(join(kp_root, cam, "*.pkl")))
        annots[cam] = [load_pkl_2d(f, conf_min=conf_min) for f in files]
    return annots


def triangulate_one_frame(kpts2d_views, Ps, min_view=2):
    valid_per_joint = (kpts2d_views[:, :, 2] > 0).sum(axis=0)
    valid_joints = np.where(valid_per_joint >= min_view)[0]
    J = kpts2d_views.shape[1]
    result = np.zeros((J, 4), dtype=np.float32)
    if len(valid_joints) == 0:
        return result
    keypoints = kpts2d_views[:, valid_joints]
    conf3d = keypoints[:, :, 2].sum(axis=0) / valid_per_joint[valid_joints]
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


def project(kp3d, P):
    valid = kp3d[:, 3] > 0
    homo = np.concatenate([kp3d[:, :3], np.ones((kp3d.shape[0], 1))], axis=1)
    proj = (P @ homo.T).T
    z = proj[:, 2]
    uv = proj[:, :2] / np.maximum(z[:, None], 1e-9)
    return uv, valid & (z > 1e-6)


def make_mixed_p(image_cam, extri_cam, cam_dict):
    """Use image camera intrinsics with candidate camera extrinsics."""
    K = cam_dict[image_cam]["K"]
    R = cam_dict[extri_cam]["R"]
    T = cam_dict[extri_cam]["T"]
    return K @ np.hstack([R, T])


def score_projection(image_cam, candidate_cam, cam_dict, annots, kp3d_by_frame,
                     frame_indices, offsets, conf_min, joints, mode):
    if mode == "full_camera":
        P = cam_dict[candidate_cam]["P"]
    elif mode == "imageK_candidateRT":
        P = make_mixed_p(image_cam, candidate_cam, cam_dict)
    else:
        raise ValueError(mode)

    errs = []
    frames_used = 0
    n_files = len(annots[image_cam])
    for fi, kp3d in zip(frame_indices, kp3d_by_frame):
        j = fi + offsets.get(image_cam, 0)
        if not (0 <= j < n_files):
            continue
        kp2d = annots[image_cam][j]
        uv, mask = project(kp3d, P)
        m = mask & (kp2d[:, 2] > conf_min)
        if joints is not None:
            joint_mask = np.zeros(kp2d.shape[0], dtype=bool)
            joint_mask[joints] = True
            m &= joint_mask
        if m.sum() < 3:
            continue
        d = np.linalg.norm(uv[m] - kp2d[m, :2], axis=1)
        errs.extend(d.tolist())
        frames_used += 1
    if not errs:
        return None
    arr = np.asarray(errs, dtype=np.float32)
    return {
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90)),
        "mean_px": float(np.mean(arr)),
        "n_joints": int(arr.size),
        "n_frames": int(frames_used),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kp_dir", default="output_2d_manual_track_full/keypoints2d")
    ap.add_argument("--intri", required=True)
    ap.add_argument("--extri", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trusted_cams", default="D03_20260325,D05_20260325,D06_20260325")
    ap.add_argument("--offsets", default=None,
                    help="optional JSON with lags, or cam:offset,cam:offset")
    ap.add_argument("--range", default="0,1249")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_triangulate_views", type=int, default=2)
    ap.add_argument("--joints", default="0,1,2,5,8,9,12")
    args = ap.parse_args()

    kp_root = join(args.root, args.kp_dir)
    cams = sorted([d for d in os.listdir(kp_root) if os.path.isdir(join(kp_root, d))])
    trusted_cams = parse_csv(args.trusted_cams)
    joints = parse_joints(args.joints)
    cam_dict = read_camera(args.intri, args.extri)
    cam_dict.pop("basenames", None)
    missing = [cam for cam in cams if cam not in cam_dict]
    if missing:
        raise ValueError(f"cameras missing from calibration: {missing}")
    annots = load_all_keypoints(kp_root, cams, conf_min=args.conf)
    offsets = parse_offsets(args.offsets, cams)

    lo, hi = [int(v) for v in args.range.split(",")]
    n_min = min(len(v) for v in annots.values())
    lo = max(lo, 0)
    hi = min(hi, n_min - 1)
    frame_indices = list(range(lo, hi + 1, args.step))

    trusted_Ps = np.stack([cam_dict[cam]["P"] for cam in trusted_cams], axis=0)
    kp3d_by_frame = []
    used_indices = []
    for fi in frame_indices:
        views = []
        valid_views = 0
        for cam in trusted_cams:
            j = fi + offsets.get(cam, 0)
            if 0 <= j < len(annots[cam]):
                kp = annots[cam][j]
                if joints is not None:
                    tmp = kp.copy()
                    keep = np.zeros(tmp.shape[0], dtype=bool)
                    keep[joints] = True
                    tmp[~keep, 2] = 0
                    kp = tmp
                if (kp[:, 2] > args.conf).sum() >= 3:
                    valid_views += 1
                views.append(kp)
            else:
                views.append(np.zeros((25, 3), dtype=np.float32))
        if valid_views < args.min_triangulate_views:
            continue
        kp3d = triangulate_one_frame(np.stack(views, axis=0), trusted_Ps,
                                     min_view=args.min_triangulate_views)
        if (kp3d[:, 3] > 0).sum() < 3:
            continue
        kp3d_by_frame.append(kp3d)
        used_indices.append(fi)

    print(f"[matrix] cams: {cams}")
    print(f"[matrix] trusted cams: {trusted_cams}")
    print(f"[matrix] frames used for 3D: {len(used_indices)} / {len(frame_indices)}")
    print(f"[matrix] joints: {joints}")
    print(f"[matrix] offsets: {offsets}")

    rows = []
    for mode in ["full_camera", "imageK_candidateRT"]:
        for image_cam in cams:
            for candidate_cam in cams:
                score = score_projection(
                    image_cam, candidate_cam, cam_dict, annots, kp3d_by_frame,
                    used_indices, offsets, args.conf, joints, mode)
                row = {
                    "mode": mode,
                    "image_cam": image_cam,
                    "candidate_cam": candidate_cam,
                    "median_px": "",
                    "p90_px": "",
                    "mean_px": "",
                    "n_frames": 0,
                    "n_joints": 0,
                }
                if score is not None:
                    row.update({
                        "median_px": round(score["median_px"], 3),
                        "p90_px": round(score["p90_px"], 3),
                        "mean_px": round(score["mean_px"], 3),
                        "n_frames": score["n_frames"],
                        "n_joints": score["n_joints"],
                    })
                rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "image_cam", "candidate_cam", "median_px", "p90_px",
            "mean_px", "n_frames", "n_joints"])
        writer.writeheader()
        writer.writerows(rows)

    # Print best candidate per image camera for quick reading.
    for mode in ["full_camera", "imageK_candidateRT"]:
        print(f"\n### {mode}")
        for image_cam in cams:
            subset = [r for r in rows if r["mode"] == mode and r["image_cam"] == image_cam and r["median_px"] != ""]
            subset = sorted(subset, key=lambda r: float(r["median_px"]))
            best = subset[0] if subset else None
            diag = next((r for r in subset if r["candidate_cam"] == image_cam), None)
            if best is None:
                print(f"{image_cam}: no data")
                continue
            diag_med = "NA" if diag is None else f"{float(diag['median_px']):.1f}"
            print(f"{image_cam}: best={best['candidate_cam']} median={float(best['median_px']):.1f}px "
                  f"diag={diag_med}px")
    print(f"\n[matrix] wrote {args.out}")


if __name__ == "__main__":
    main()
