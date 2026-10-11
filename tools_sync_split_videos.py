"""Sync-aware splitting of long multi-camera videos.

Workflow:
  1. Identify a per-camera time offset (sync_offset_sec) — either from a
     known reference event (visual clap, light flash, etc.) or from a
     prior sync search (lags.json frame-indices converted to seconds).
  2. Choose a global time window [t_start, t_end] in the synchronized
     timeline.
  3. For each camera C, extract the segment [t_start + offset_C, t_end + offset_C]
     from its raw video. The resulting 8 clips are frame-aligned.
  4. Optionally extract each segment to JPG frames at a configurable rate.

Usage example:
  python tools_sync_split_videos.py \
      --sync_json sync_offsets.json \
      --t_start 600 --t_end 660 \
      --output_dir aligned_clips \
      --fps_out 10 --extract_frames

sync_offsets.json format:
{
  "fps": 50,
  "offsets_sec": {
      "D01_xxx": 0.14,
      "D02_xxx": 0.00,
      ...
  },
  "videos": {
      "D01_xxx": "/path/to/D01.mp4",
      ...
  }
}
"""

import argparse
import json
import os
import subprocess
from os.path import join


def run(cmd, check=True):
    print(' '.join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def probe_video(path):
    """Get fps and duration via ffprobe."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate,duration,nb_frames',
           '-of', 'json', path]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(r.stdout)['streams'][0]
    num, den = info['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    duration = float(info.get('duration', 0))
    return fps, duration


def extract_segment(video_path, t_start, t_end, out_path):
    """Cut a sub-clip [t_start, t_end] from video_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ['ffmpeg', '-y',
           '-ss', f'{t_start:.3f}',
           '-i', video_path,
           '-t', f'{t_end - t_start:.3f}',
           '-c:v', 'libx264',
           '-preset', 'fast',
           '-crf', '18',
           '-an',
           out_path]
    run(cmd)


def extract_frames(video_path, out_dir, fps_out):
    """Decode a sub-clip to JPG frames at fps_out."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = ['ffmpeg', '-y',
           '-i', video_path,
           '-vf', f'fps={fps_out}',
           '-q:v', '2',
           '-start_number', '0',
           join(out_dir, '%06d.jpg')]
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync_json', required=True,
                    help='JSON with {fps, offsets_sec, videos}')
    ap.add_argument('--t_start', type=float, required=True,
                    help='global start time (seconds, in synced timeline)')
    ap.add_argument('--t_end', type=float, required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--extract_frames', action='store_true')
    ap.add_argument('--fps_out', type=float, default=10.0,
                    help='frames per second when extracting JPGs')
    args = ap.parse_args()

    with open(args.sync_json) as f:
        sync = json.load(f)
    offsets = sync['offsets_sec']
    videos = sync['videos']

    print(f'[sync] global window [{args.t_start}, {args.t_end}] sec')
    print(f'[sync] {len(videos)} cameras')

    os.makedirs(args.output_dir, exist_ok=True)

    for cam, video_path in videos.items():
        offset = offsets.get(cam, 0.0)
        # camera local time = global time + per-camera offset
        cam_t_start = args.t_start + offset
        cam_t_end = args.t_end + offset
        print(f'\n[{cam}] offset={offset:+.3f}s -> '
              f'extract [{cam_t_start:.3f}, {cam_t_end:.3f}]')

        # Probe to make sure window is inside video
        fps, dur = probe_video(video_path)
        if cam_t_end > dur:
            print(f'  WARNING: window exceeds video duration ({dur:.1f}s) -- '
                  f'will be truncated')

        clip_path = join(args.output_dir, 'videos', f'{cam}.mp4')
        extract_segment(video_path, cam_t_start, cam_t_end, clip_path)

        if args.extract_frames:
            frames_dir = join(args.output_dir, 'images', cam)
            extract_frames(clip_path, frames_dir, args.fps_out)
            n = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            print(f'  {n} frames in {frames_dir}')

    print(f'\n[done] output -> {args.output_dir}')


if __name__ == '__main__':
    main()
