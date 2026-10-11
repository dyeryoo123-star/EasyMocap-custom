"""Detect the frame at which the on-screen second timestamp rolls over.

The videos have a date/time overlay in the top-left corner with second-
level precision (e.g. "01/04/2026 17:01:46"). The exact moment the
second number changes from N to N+1 is precise to one frame interval
and gives a sync anchor across cameras.

Strategy:
  1. Crop the top-left timestamp region from each frame.
  2. OCR or compare-against-previous-frame to detect when the
     timestamp pixels change.
  3. Report the frame index of the first change after a given start
     time.

Two modes:
  - 'change': fast, no OCR. Detects the frame where the timestamp ROI
    pixel content changes significantly (= second rolled over).
  - 'ocr': uses easyocr/tesseract to actually read the time string.
    Slower but tells you the exact second value.

Usage:
  python tools_detect_timestamp_rollover.py \
      --videos D01.mp4 D02.mp4 ... D08.mp4 \
      --probe_start 0 --probe_duration 5 \
      --roi_xywh 0 0 400 60 \
      --mode change \
      --save_json /tmp/rollover_offsets.json
"""

import argparse
import json
import os
import subprocess
from os.path import basename

import cv2
import numpy as np


def get_video_fps(path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate', '-of', 'json', path]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    num, den = json.loads(r.stdout)['streams'][0]['r_frame_rate'].split('/')
    return float(num) / float(den)


def detect_rollover_by_change(video_path, probe_start_sec, probe_duration_sec,
                                 roi_xywh, change_thresh=15.0):
    """Find frame index where timestamp ROI changes significantly. The
    first change after probe_start is the rollover frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f = int(probe_start_sec * fps)
    end_f = int((probe_start_sec + probe_duration_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    x, y, w, h = roi_xywh
    prev_roi = None
    rollover_frame = None
    diffs = []
    for i in range(start_f, end_f):
        ok, frame = cap.read()
        if not ok:
            break
        roi = frame[y:y+h, x:x+w].copy()
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if prev_roi is None:
            prev_roi = roi_gray
            continue
        # Compare pixel diff to detect timestamp change
        diff = float(np.abs(roi_gray.astype(int) - prev_roi.astype(int)).mean())
        diffs.append((i, diff))
        if diff > change_thresh and rollover_frame is None:
            rollover_frame = i
            # Don't break — keep recording diffs to validate
        prev_roi = roi_gray
    cap.release()
    return rollover_frame, fps, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--probe_start', type=float, default=0.0,
                    help='start probing at this many seconds into the video')
    ap.add_argument('--probe_duration', type=float, default=3.0,
                    help='probe this many seconds — must include exactly one '
                         'second-rollover (so >=2.0)')
    ap.add_argument('--roi_xywh', nargs=4, type=int, default=[0, 0, 500, 70],
                    help='timestamp ROI: x y w h pixels (top-left default)')
    ap.add_argument('--change_thresh', type=float, default=15.0,
                    help='pixel-diff threshold to flag a change')
    ap.add_argument('--save_json', default=None)
    args = ap.parse_args()

    results = {}
    fpss = {}
    print(f'Probing each video at [{args.probe_start},'
          f' {args.probe_start + args.probe_duration}]s, ROI={args.roi_xywh}')
    print('Looking for the first second-rollover within that window.\n')

    for v in args.videos:
        cam = basename(v).split('.')[0]
        rollover, fps, diffs = detect_rollover_by_change(
            v, args.probe_start, args.probe_duration, args.roi_xywh,
            args.change_thresh)
        fpss[cam] = fps
        if rollover is None:
            print(f'[{cam}] NO ROLLOVER FOUND. fps={fps:.1f}. '
                  f'Try wider window or different ROI.')
            top = sorted(diffs, key=lambda x: -x[1])[:3]
            print(f'  top-3 frame diffs: {top}')
        else:
            results[cam] = rollover
            t_rollover = rollover / fps
            print(f'[{cam}] rollover at frame {rollover} (t={t_rollover:.3f}s), fps={fps:.1f}')

    if not results:
        print('\nNo rollovers detected. Adjust --probe_start / --probe_duration / --roi_xywh.')
        return

    # Compute offsets relative to first cam
    ref_cam = list(results.keys())[0]
    ref_frame = results[ref_cam]
    ref_t = ref_frame / fpss[ref_cam]
    print(f'\nReference: {ref_cam} at frame {ref_frame} (t={ref_t:.3f}s)\n')
    offsets_frame = {}
    offsets_sec = {}
    for cam, frame in results.items():
        df = frame - ref_frame
        dt = frame / fpss[cam] - ref_t
        offsets_frame[cam] = df
        offsets_sec[cam] = dt
        print(f'  {cam}: offset = {df:+d} frames ({dt:+.3f}s)')

    out = {
        'method': 'timestamp_rollover',
        'reference': ref_cam,
        'reference_frame': ref_frame,
        'fps': fpss,
        'rollover_frame': results,
        'offsets_frames': offsets_frame,
        'offsets_sec': offsets_sec,
    }
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nsaved -> {args.save_json}')


if __name__ == '__main__':
    main()
