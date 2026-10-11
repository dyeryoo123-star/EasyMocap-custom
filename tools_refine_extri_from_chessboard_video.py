"""Refine chessboard-origin extrinsics from the static board visible in videos.

This script is intentionally small and reproducible:

1. Read EasyMocap-style intrinsics and an optional previous extri file.
2. Detect the 6x4-square board as a 5x3 inner-corner chessboard.
3. Run per-camera solvePnP with square size in metres.
4. Resolve chessboard corner-order ambiguity using the previous extrinsics
   as a soft prior, then write a new EasyMocap-compatible extri.yml.

For this project the board has 6x4 squares, so OpenCV's internal-corner
pattern is 5x3 and the grid size is 0.19 m.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from easymocap.mytools.camera_utils import FileStorage, write_extri


@dataclass
class Candidate:
    cam: str
    frame: int
    pattern: Tuple[int, int]
    method: str
    scale: float
    variant: str
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float
    old_center_dist_m: Optional[float]
    old_rot_angle_deg: Optional[float]
    score: float
    bbox: Tuple[float, float, float, float]
    area_ratio: float
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    corners: np.ndarray
    obj: np.ndarray
    image_path: Optional[str] = None

    def to_report(self) -> Dict:
        out = asdict(self)
        for key in ("rvec", "tvec", "R", "corners", "obj"):
            out.pop(key, None)
        out["pattern"] = list(self.pattern)
        out["bbox"] = list(self.bbox)
        return out


def parse_pattern(text: str) -> Tuple[int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"Pattern must look like 5x3, got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_frames(text: str) -> List[int]:
    frames: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            if len(bits) == 2:
                start, end = bits
                step = 1
            elif len(bits) == 3:
                start, end, step = bits
            else:
                raise ValueError(f"Bad frame range {part!r}")
            frames.extend(range(start, end + 1, step))
        else:
            frames.append(int(part))
    return sorted(set(frames))


def parse_overrides(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not text:
        return out
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        cam, frame = item.split("=", 1)
        out[cam.strip()] = int(frame)
    return out


def read_intri(path: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    fs = FileStorage(path)
    names = fs.read("names", dt="list")
    intri = {}
    for cam in names:
        intri[cam] = {
            "K": fs.read(f"K_{cam}"),
            "dist": fs.read(f"dist_{cam}"),
        }
    return intri, names


def read_extri(path: Optional[str]) -> Dict[str, Dict[str, np.ndarray]]:
    if not path or not os.path.exists(path):
        return {}
    fs = FileStorage(path)
    names = fs.read("names", dt="list")
    extri = {}
    for cam in names:
        rvec = fs.read(f"R_{cam}")
        R = fs.read(f"Rot_{cam}")
        T = fs.read(f"T_{cam}")
        if R is None and rvec is not None:
            R = cv2.Rodrigues(rvec)[0]
        extri[cam] = {"Rvec": rvec, "R": R, "T": T}
    return extri


def camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (-R.T @ T.reshape(3, 1)).reshape(3)


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra @ Rb.T
    val = (np.trace(R) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def build_object_points(pattern: Tuple[int, int], grid: float, variant: str) -> np.ndarray:
    cols, rows = pattern
    base = np.zeros((rows, cols, 3), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            base[y, x, 0] = x * grid
            base[y, x, 1] = y * grid
    if variant == "identity":
        arr = base
    elif variant == "flip_x":
        arr = base[:, ::-1]
    elif variant == "flip_y":
        arr = base[::-1, :]
    elif variant == "flip_xy":
        arr = base[::-1, ::-1]
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    return arr.reshape(-1, 3).astype(np.float32)


def detect_chessboard_candidates(gray: np.ndarray,
                                 pattern: Tuple[int, int],
                                 scales: Iterable[float]) -> List[Tuple[Tuple[int, int], np.ndarray, str, float]]:
    """Collect detection candidates instead of trusting the first hit.

    SB is powerful but can occasionally lock onto a false 5x3 pattern on a
    high-contrast board/background. Keeping all method/scale candidates lets
    solvePnP reprojection error reject those false positives.
    """
    found: List[Tuple[Tuple[int, int], np.ndarray, str, float]] = []
    patterns = [pattern]
    for scale in scales:
        g = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for pat in patterns:
            try:
                ok, corners = cv2.findChessboardCorners(
                    g, pat,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                    | cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_FAST_CHECK,
                )
            except cv2.error:
                ok, corners = False, None
            if ok and corners is not None:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
                corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), crit)
                pts = corners.reshape(-1, 2).astype(np.float32)
                if scale != 1.0:
                    pts /= scale
                found.append((pat, pts, "classic", scale))
            for flags in (
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            ):
                try:
                    ok, corners = cv2.findChessboardCornersSB(g, pat, flags=flags)
                except cv2.error:
                    ok, corners = False, None
                if ok and corners is not None:
                    pts = corners.reshape(-1, 2).astype(np.float32)
                    if scale != 1.0:
                        pts /= scale
                    found.append((pat, pts, "SB", scale))
    return found


def solve_candidate(cam: str,
                    frame: int,
                    image: np.ndarray,
                    pattern: Tuple[int, int],
                    method: str,
                    scale: float,
                    corners: np.ndarray,
                    variant: str,
                    grid: float,
                    K: np.ndarray,
                    dist: np.ndarray,
                    old: Optional[Dict[str, np.ndarray]],
                    prior_center_weight: float,
                    prior_rot_weight: float) -> Optional[Candidate]:
    obj = build_object_points(pattern, grid, variant)
    flags = cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=flags)
    if not ok:
        return None
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, corners, K, dist, rvec, tvec)
    except cv2.error:
        pass
    R = cv2.Rodrigues(rvec)[0]
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners, axis=1)
    xs, ys = corners[:, 0], corners[:, 1]
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    area_ratio = float(area / (image.shape[0] * image.shape[1]))
    old_center_dist = None
    old_rot_angle = None
    score = float(np.mean(err))
    if old is not None and old.get("R") is not None and old.get("T") is not None:
        c_new = camera_center(R, tvec)
        c_old = camera_center(old["R"], old["T"])
        old_center_dist = float(np.linalg.norm(c_new - c_old))
        old_rot_angle = rotation_angle_deg(R, old["R"])
        score += prior_center_weight * old_center_dist
        score += prior_rot_weight * (old_rot_angle / 180.0)
    return Candidate(
        cam=cam,
        frame=frame,
        pattern=pattern,
        method=method,
        scale=scale,
        variant=variant,
        reproj_mean_px=float(np.mean(err)),
        reproj_median_px=float(np.median(err)),
        reproj_max_px=float(np.max(err)),
        old_center_dist_m=old_center_dist,
        old_rot_angle_deg=old_rot_angle,
        score=float(score),
        bbox=bbox,
        area_ratio=area_ratio,
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        R=R,
        corners=corners,
        obj=obj,
    )


def get_video_path(video_dir: Path, cam: str) -> Optional[Path]:
    prefix = cam.split("_")[0]
    matches = sorted(video_dir.glob(f"{prefix}_*.mp4"))
    return matches[0] if matches else None


def read_frame(video: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_overlay(image: np.ndarray, cand: Candidate) -> np.ndarray:
    out = image.copy()
    proj, _ = cv2.projectPoints(cand.obj, cand.rvec, cand.tvec, np.eye(3), np.zeros((1, 5)))
    # The projection above is intentionally not used for drawing because K/dist
    # are not stored in Candidate; caller draws with draw_projected below.
    return out


def draw_projected(image: np.ndarray,
                   cand: Candidate,
                   K: np.ndarray,
                   dist: np.ndarray) -> np.ndarray:
    out = image.copy()
    proj, _ = cv2.projectPoints(cand.obj, cand.rvec, cand.tvec, K, dist)
    proj = proj.reshape(-1, 2)
    for i, (uv, q) in enumerate(zip(cand.corners, proj)):
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        x, y = int(round(float(q[0]))), int(round(float(q[1])))
        cv2.circle(out, (u, v), 7, (0, 255, 255), 2)
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
        if i in (0, len(proj) - 1):
            cv2.putText(out, str(i), (u + 6, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    axis = np.array([[0, 0, 0], [0.38, 0, 0], [0, 0.38, 0], [0, 0, -0.38]],
                    dtype=np.float32)
    ax, _ = cv2.projectPoints(axis, cand.rvec, cand.tvec, K, dist)
    ax = ax.reshape(-1, 2).astype(int)
    o = tuple(ax[0])
    cv2.line(out, o, tuple(ax[1]), (0, 0, 255), 4)
    cv2.line(out, o, tuple(ax[2]), (0, 255, 0), 4)
    cv2.line(out, o, tuple(ax[3]), (255, 0, 0), 4)
    cv2.putText(out, f"{cand.cam[:3]} f={cand.frame} err={cand.reproj_mean_px:.2f}px {cand.variant}",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    return out


def make_contact_sheet(images: List[np.ndarray], out_path: Path) -> None:
    if not images:
        return
    tiles = []
    for img in images:
        tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    while len(tiles) < 8:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--intri", required=True)
    parser.add_argument("--extri_prior", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_name", default="extri_chess_refined.yml")
    parser.add_argument("--pattern", default="5x3")
    parser.add_argument("--grid", type=float, default=0.19)
    parser.add_argument("--frames", default="0,25,50,100,150,250,500,750,1000")
    parser.add_argument("--frame_overrides", default="")
    parser.add_argument("--scales", default="1.0,1.5,2.0")
    parser.add_argument("--max_reproj_px", type=float, default=8.0)
    parser.add_argument("--prior_center_weight", type=float, default=1.0,
                        help="Score penalty per metre away from prior camera centre.")
    parser.add_argument("--prior_rot_weight", type=float, default=3.0,
                        help="Score penalty for 180 degrees away from prior rotation.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    pattern = parse_pattern(args.pattern)
    frames = parse_frames(args.frames)
    overrides = parse_overrides(args.frame_overrides)
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    intri, names = read_intri(args.intri)
    prior = read_extri(args.extri_prior)

    selected: Dict[str, Candidate] = {}
    all_rows: List[Dict] = []
    overlay_images: List[np.ndarray] = []

    for cam in names:
        video = get_video_path(Path(args.video_dir), cam)
        if video is None:
            print(f"[{cam}] no matching video")
            continue
        cam_frames = [overrides[cam[:3]]] if cam[:3] in overrides else frames
        candidates: List[Candidate] = []
        for frame_idx in cam_frames:
            image = read_frame(video, frame_idx)
            if image is None:
                print(f"[{cam}] failed to read frame {frame_idx}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = detect_chessboard_candidates(gray, pattern, scales)
            if not detections:
                print(f"[{cam}] frame {frame_idx}: no chessboard")
                continue
            for det_pattern, corners, method, scale in detections:
                for variant in ("identity", "flip_x", "flip_y", "flip_xy"):
                    cand = solve_candidate(
                        cam=cam,
                        frame=frame_idx,
                        image=image,
                        pattern=det_pattern,
                        method=method,
                        scale=scale,
                        corners=corners,
                        variant=variant,
                        grid=args.grid,
                        K=intri[cam]["K"],
                        dist=intri[cam]["dist"],
                        old=prior.get(cam),
                        prior_center_weight=args.prior_center_weight,
                        prior_rot_weight=args.prior_rot_weight,
                    )
                    if cand is not None:
                        candidates.append(cand)
                        all_rows.append(cand.to_report())
        if not candidates:
            print(f"[{cam}] no usable candidates")
            continue
        candidates.sort(key=lambda c: c.score)
        chosen = candidates[0]
        selected[cam] = chosen
        image = read_frame(video, chosen.frame)
        if image is not None:
            overlay = draw_projected(image, chosen, intri[cam]["K"], intri[cam]["dist"])
            overlay_path = overlay_dir / f"{cam}_frame{chosen.frame:06d}.jpg"
            cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            chosen.image_path = str(overlay_path)
            overlay_images.append(overlay)
        status = "OK" if chosen.reproj_mean_px <= args.max_reproj_px else "CHECK"
        print(
            f"[{cam}] {status} frame={chosen.frame} method={chosen.method}@{chosen.scale:g} "
            f"variant={chosen.variant} err={chosen.reproj_mean_px:.3f}px "
            f"prior_center={chosen.old_center_dist_m if chosen.old_center_dist_m is not None else -1:.3f}m "
            f"prior_rot={chosen.old_rot_angle_deg if chosen.old_rot_angle_deg is not None else -1:.1f}deg"
        )

    if not selected:
        raise RuntimeError("No camera extrinsics were solved.")

    # If one camera fails, keep its prior instead of silently dropping it.
    cameras_out: Dict[str, Dict[str, np.ndarray]] = {}
    for cam in names:
        if cam in selected and selected[cam].reproj_mean_px <= args.max_reproj_px:
            cand = selected[cam]
            cameras_out[cam] = {"Rvec": cand.rvec, "R": cand.R, "T": cand.tvec}
        elif cam in prior:
            reason = "refinement failed"
            if cam in selected:
                reason = f"candidate reprojection is {selected[cam].reproj_mean_px:.2f}px > {args.max_reproj_px:.2f}px"
            print(f"[{cam}] WARNING: using prior extrinsic because {reason}")
            cameras_out[cam] = {
                "Rvec": prior[cam]["Rvec"],
                "R": prior[cam]["R"],
                "T": prior[cam]["T"],
            }

    out_yml = out_dir / args.out_name
    write_extri(str(out_yml), cameras_out)

    selected_report = {cam: cand.to_report() for cam, cand in selected.items()}
    for cam, cand in selected.items():
        if cand.image_path:
            selected_report[cam]["overlay"] = cand.image_path
    report = {
        "pattern_inner_corners": list(pattern),
        "square_size_m": args.grid,
        "video_dir": str(Path(args.video_dir)),
        "intri": args.intri,
        "extri_prior": args.extri_prior,
        "out_yml": str(out_yml),
        "selected": selected_report,
    }
    with open(out_dir / "extri_chess_refined_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if all_rows:
        with open(out_dir / "extri_chess_refined_candidates.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    make_contact_sheet(overlay_images, out_dir / "extri_chess_refined_overlay_contact.jpg")
    print(f"\n[wrote] {out_yml}")
    print(f"[wrote] {out_dir / 'extri_chess_refined_report.json'}")
    print(f"[wrote] {out_dir / 'extri_chess_refined_candidates.csv'}")
    print(f"[wrote] {out_dir / 'extri_chess_refined_overlay_contact.jpg'}")


if __name__ == "__main__":
    main()
