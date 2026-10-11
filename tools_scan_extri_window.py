"""Scan SmallSpaceA videos to find time windows where the 6x4 (= 5x3
internal corner) chessboard is visible to many cameras simultaneously.

For each candidate wall-clock time:
  - extract the synchronized frame from each cam (via sync_meta.json)
  - run chessboard detection on each frame
  - record n_cams_with_detection and corner spread

Outputs:
  - <out_dir>/scan_TIME.jpg: 2x4 mosaic of all 8 cams at that time,
    with green box around detected chessboards
  - <out_dir>/scan_report.csv: row per time with detection counts
"""

import argparse
import json
import os
import subprocess
from os.path import join

import cv2
import numpy as np


def parse_time(s):
    parts = s.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def fmt_time(sec_since_midnight):
    h = int(sec_since_midnight // 3600)
    m = int((sec_since_midnight % 3600) // 60)
    s = sec_since_midnight % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def detect(gray, pattern):
    """Try multiple detection strategies. Returns first successful corners
    or None. Patterns (a,b) and (b,a) are both tried because the board
    can appear in either orientation."""
    pat_alt = (pattern[1], pattern[0])
    # 1. classic with full flags
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
            return c.reshape(-1, 2)
    # 2. SB detector (catches D06/D08-style hard cases)
    for pat in (pattern, pat_alt):
        for flags in (cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY,
                      cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE):
            try:
                ok, c = cv2.findChessboardCornersSB(gray, pat, flags=flags)
            except cv2.error:
                ok = False
            if ok and c is not None:
                return c.reshape(-1, 2)
    return None


def extract_frame(video, t_in_video_sec, max_height=540):
    """Extract one frame as numpy array via ffmpeg."""
    cmd = ['ffmpeg', '-y', '-ss', f'{t_in_video_sec:.3f}',
           '-i', video, '-vframes', '1', '-vf',
           f'scale=-2:{max_height}', '-f', 'image2pipe',
           '-vcodec', 'mjpeg', 'pipe:1']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    arr = np.frombuffer(r.stdout, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync_meta', required=True)
    ap.add_argument('--t_start', default='13:03:00')
    ap.add_argument('--t_end',   default='14:09:00')
    ap.add_argument('--step_sec', type=float, default=300.0,
                    help='scan one frame every N seconds (default 300=5min)')
    ap.add_argument('--pattern', default='5x3')
    ap.add_argument('--out_dir', default='/tmp/extri_scan')
    ap.add_argument('--max_h', type=int, default=540,
                    help='thumbnail height for mosaic')
    args = ap.parse_args()

    pat = tuple(int(x) for x in args.pattern.split('x'))
    meta = json.load(open(args.sync_meta))
    cams = list(meta['cameras'].items())

    t0 = parse_time(args.t_start)
    t1 = parse_time(args.t_end)
    os.makedirs(args.out_dir, exist_ok=True)

    csv_lines = ['wallclock,n_detected,cams_detected,detected_cams']
    print(f"scanning [{args.t_start}, {args.t_end}] every {args.step_sec}s")
    print(f"{'time':<14}{'n_det':>6}  detected")

    t = t0
    while t <= t1:
        wall = fmt_time(t)
        tiles = []
        n_det = 0
        det_cams = []
        for cam, info in cams:
            t_in_v = t - info['start_sec_since_midnight']
            if t_in_v < 0 or t_in_v > info['duration_sec']:
                tile = np.zeros((args.max_h, args.max_h * 16 // 9, 3), np.uint8)
            else:
                img = extract_frame(info['video'], t_in_v, args.max_h)
                if img is None:
                    tile = np.zeros((args.max_h, args.max_h * 16 // 9, 3),
                                    np.uint8)
                else:
                    tile = img
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    c = detect(gray, pat)
                    if c is not None:
                        xs, ys = c[:, 0], c[:, 1]
                        cv2.rectangle(tile, (int(xs.min()), int(ys.min())),
                                      (int(xs.max()), int(ys.max())),
                                      (0, 255, 0), 3)
                        n_det += 1
                        det_cams.append(cam[:3])
            label = f'{cam[:3]} t_in_v={t_in_v:.1f}s'
            cv2.putText(tile, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tiles.append(cv2.resize(tile, (480, 270)))
        # 2x4 grid
        row1 = np.hstack(tiles[:4])
        row2 = np.hstack(tiles[4:8])
        mosaic = np.vstack([row1, row2])
        cv2.putText(mosaic, f'{wall}  detected={n_det}/8 ({",".join(det_cams)})',
                    (15, 530), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        out = join(args.out_dir, f'scan_{wall.replace(":","")}.jpg')
        cv2.imwrite(out, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 78])
        print(f'{wall:<14}{n_det:>6}  {",".join(det_cams)}')
        csv_lines.append(f'{wall},{n_det},{len(det_cams)},'
                         f'"{",".join(det_cams)}"')
        t += args.step_sec

    with open(join(args.out_dir, 'scan_report.csv'), 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f'\nreport -> {join(args.out_dir, "scan_report.csv")}')
    print(f'mosaics -> {args.out_dir}/scan_*.jpg')


if __name__ == '__main__':
    main()
