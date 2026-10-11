"""Fast extraction of time-aligned multi-camera frames.

This is intended for long videos with mixed source frame rates. It uses a
shared wall-clock timeline from sync_meta.json, seeks each camera to the
corresponding local video time, and writes the same number of output frames
for every camera at a fixed output FPS.

Example:
  python tools_extract_aligned_segment_fast.py \
      --sync_meta outputs/smallspacea_dingyu_1300_1400/sync_meta_regenerated.json \
      --t_start 13:15:25 \
      --t_end 13:16:00 \
      --output_fps 25 \
      --out_dir outputs/smallspacea_dingyu_1300_1400/mocap_seg_131525_131600_fast
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_time(s):
    """Parse HH:MM:SS or HH:MM:SS.sss into seconds since midnight."""
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def clean_generated_jpgs(cam_dir):
    """Remove only generated JPGs in the target camera folder."""
    if not cam_dir.exists():
        return
    for jpg in cam_dir.glob("*.jpg"):
        jpg.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync_meta", required=True)
    ap.add_argument("--t_start", required=True, help="global wall-clock start, e.g. 13:15:25")
    ap.add_argument("--t_end", required=True, help="global wall-clock end, e.g. 13:16:00")
    ap.add_argument("--output_fps", type=float, default=25.0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--jpeg_quality", type=int, default=2)
    args = ap.parse_args()

    meta = json.load(open(args.sync_meta))
    cams = meta["cameras"]
    t_start = parse_time(args.t_start)
    t_end = parse_time(args.t_end)
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError("--t_end must be after --t_start")

    n_frames = int(round(duration * args.output_fps))
    out_dir = Path(args.out_dir)
    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "sync_meta": os.path.abspath(args.sync_meta),
        "method": "ffmpeg seek per camera + fps filter + fixed frame count",
        "t_start": args.t_start,
        "t_end": args.t_end,
        "duration_sec": duration,
        "output_fps": args.output_fps,
        "frames_per_camera": n_frames,
        "cameras": {},
    }

    print(f"time window: {args.t_start} -> {args.t_end} ({duration:.3f}s)")
    print(f"output fps:  {args.output_fps}")
    print(f"frames/cam:  {n_frames}")

    for cam, info in sorted(cams.items()):
        cam_dir = images_root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        clean_generated_jpgs(cam_dir)

        t_in_video = t_start - float(info["start_sec_since_midnight"])
        if t_in_video < 0:
            raise ValueError(f"{cam}: requested start is before this camera starts")

        duration_available = float(info["duration_sec"]) - t_in_video
        requested_duration = n_frames / args.output_fps
        if duration_available < requested_duration:
            raise ValueError(
                f"{cam}: requested segment exceeds video duration "
                f"(available {duration_available:.3f}s, requested {requested_duration:.3f}s)"
            )

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{t_in_video:.6f}",
            "-i", info["video"],
            "-vf", f"fps={args.output_fps}",
            "-frames:v", str(n_frames),
            "-q:v", str(args.jpeg_quality),
            "-start_number", "0",
            str(cam_dir / "%06d.jpg"),
        ]
        subprocess.run(cmd, check=True)
        files = sorted(cam_dir.glob("*.jpg"))
        plan["cameras"][cam] = {
            "video": info["video"],
            "camera_start_wallclock": info["start_wallclock"],
            "t_in_video_sec": round(t_in_video, 6),
            "source_fps": info["fps"],
            "frames_written": len(files),
            "first_output": files[0].name if files else None,
            "last_output": files[-1].name if files else None,
        }
        print(f"{cam}: t_in={t_in_video:.3f}s -> {len(files)} frames")

    with open(out_dir / "extraction_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
