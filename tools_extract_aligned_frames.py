"""Extract time-synchronized frames from 8 cameras using sync_meta.json.

For each cam C with start time T_C and actual fps f_C, the frame at
wall-clock time t is index round((t - T_C) * f_C). This tool walks a
user-specified time window at a user-specified output rate, computes
each cam's frame index, and extracts those frames to disk with shared
output indexing.

Output structure (mocap-pipeline-compatible):
  out_dir/
    images/
      D01_xxx/000000.jpg ... NNNNNN.jpg
      D02_xxx/000000.jpg ... NNNNNN.jpg
      ...

For each output index i, the file <cam>/<i>.jpg is the cam's frame
closest to wall-clock time (t_start + i / output_fps).

Usage:
  python tools_extract_aligned_frames.py \
      --sync_meta sync_meta.json \
      --t_start 13:03:00 \
      --t_end   13:08:00 \
      --output_fps 25 \
      --out_dir aligned_segments/seg_13_03_to_13_08
"""

import argparse
import json
import os
import subprocess
from os.path import join, basename


def parse_time(s):
    """'13:03:00' or '13:03:00.500' → seconds since midnight."""
    parts = s.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def extract_frame_at_time(video_path, t_in_video_sec, out_path):
    """Extract a single frame at given time-in-video using ffmpeg -ss.
    This uses keyframe-accurate seeking when -ss is before -i, and
    frame-accurate when after -i."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ['ffmpeg', '-y',
           '-ss', f'{t_in_video_sec:.4f}',
           '-i', video_path,
           '-vframes', '1',
           '-q:v', '2',
           out_path]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_segment_frames_via_filter(video_path, frame_indices, out_dir):
    """More efficient: use ffmpeg's select filter to extract many specific
    frame indices in one pass. Output files numbered sequentially."""
    os.makedirs(out_dir, exist_ok=True)
    if not frame_indices:
        return
    select_expr = '+'.join(f'eq(n\\,{f})' for f in frame_indices)
    cmd = ['ffmpeg', '-y',
           '-i', video_path,
           '-vf', f'select={select_expr}',
           '-vsync', '0',
           '-q:v', '2',
           '-start_number', '0',
           join(out_dir, '%06d.jpg')]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync_meta', required=True)
    ap.add_argument('--t_start', required=True,
                    help='wall-clock start time, e.g. 13:03:00')
    ap.add_argument('--t_end', required=True,
                    help='wall-clock end time, e.g. 13:08:00')
    ap.add_argument('--output_fps', type=float, default=25.0,
                    help='output sampling rate; cap at slowest cam fps')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--method', choices=['per-frame', 'filter'],
                    default='filter',
                    help='per-frame: one ffmpeg call per frame (slow but '
                         'reliable). filter: one ffmpeg call per cam '
                         '(fast).')
    args = ap.parse_args()

    meta = json.load(open(args.sync_meta))
    cams = meta['cameras']

    t_start_w = parse_time(args.t_start)
    t_end_w = parse_time(args.t_end)
    duration = t_end_w - t_start_w
    if duration <= 0:
        raise ValueError('t_end must be after t_start')
    dt = 1.0 / args.output_fps
    n_out_frames = int(round(duration * args.output_fps))
    print(f'Time window: {args.t_start} → {args.t_end} ({duration:.2f}s)')
    print(f'Output FPS:  {args.output_fps} → {n_out_frames} output frames')
    print(f'Cameras:     {len(cams)}')
    print()

    os.makedirs(join(args.out_dir, 'images'), exist_ok=True)

    # Pre-compute, per cam, the list of (out_idx, src_frame, src_time_sec)
    cam_jobs = {}
    for cam, info in cams.items():
        t_cam_start_w = parse_time(info['start_wallclock'])
        fps = info['fps']
        plan = []
        for i in range(n_out_frames):
            t_global = t_start_w + i * dt
            t_in_video = t_global - t_cam_start_w
            if t_in_video < 0:
                continue  # before cam started
            if t_in_video > info['duration_sec']:
                continue  # after cam ended
            src_frame = int(round(t_in_video * fps))
            plan.append((i, src_frame, t_in_video))
        cam_jobs[cam] = (info, plan)
        print(f'  {cam[:3]}: fps={fps:.2f}, plan {len(plan)} frames '
              f'(src frames {plan[0][1]}..{plan[-1][1]} if any)')

    # Save the plan for reproducibility
    plan_path = join(args.out_dir, 'extraction_plan.json')
    plan_dump = {
        'time_window': [args.t_start, args.t_end],
        'output_fps': args.output_fps,
        'n_output_frames': n_out_frames,
        'cameras': {
            cam: [{'out_idx': p[0], 'src_frame': p[1],
                   't_in_video': round(p[2], 4)} for p in jobs[1]]
            for cam, jobs in cam_jobs.items()
        },
    }
    with open(plan_path, 'w') as f:
        json.dump(plan_dump, f, indent=2)
    print(f'\nSaved plan to {plan_path}')

    # Extract
    print('\nExtracting...')
    for cam, (info, plan) in cam_jobs.items():
        cam_out = join(args.out_dir, 'images', cam)
        os.makedirs(cam_out, exist_ok=True)
        if args.method == 'filter':
            indices = [p[1] for p in plan]
            extract_segment_frames_via_filter(info['video'], indices, cam_out)
            # Rename sequential outputs to match out_idx (in case some cams
            # had different starting indices)
            sequential_files = sorted([f for f in os.listdir(cam_out)
                                        if f.endswith('.jpg')])
            for (out_idx, _, _), seq_file in zip(plan, sequential_files):
                seq_path = join(cam_out, seq_file)
                target = join(cam_out, f'{out_idx:06d}.jpg')
                if seq_path != target:
                    os.rename(seq_path, target)
        else:
            for (out_idx, src_frame, t_in_video) in plan:
                out_path = join(cam_out, f'{out_idx:06d}.jpg')
                extract_frame_at_time(info['video'], t_in_video, out_path)
        print(f'  {cam[:3]}: {len(plan)} frames -> {cam_out}')

    print(f'\nDone. Output: {args.out_dir}')


if __name__ == '__main__':
    main()
