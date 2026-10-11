"""Helper to find a common synchronization event across 8 camera videos.

Two modes:

(a) Audio-based: if videos have audio, detect a sharp transient (clap,
    bang) — usually the highest peak in the first N seconds is the sync
    event. The peak time per camera gives the offset directly.

(b) Visual-based: extract a thumbnail strip every K frames from each
    video around a candidate time window, save as a single image per
    camera. Open them side-by-side and manually identify the same
    event (e.g., person walks in, light flashes). Note the frame index
    of the event in each, compute deltas.

Usage:
  # Audio mode (works if videos have audio)
  python tools_find_sync_event.py --mode audio --videos D01.mp4 D02.mp4 ...

  # Visual thumbnail mode
  python tools_find_sync_event.py --mode visual \
      --videos D01.mp4 D02.mp4 ... \
      --probe_start 0 --probe_duration 30 --thumb_every_sec 0.5 \
      --out_dir /tmp/sync_thumbs
"""

import argparse
import json
import os
import subprocess
from os.path import join, basename


def run_q(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------- Audio mode ----------

def detect_audio_peak(video_path, probe_seconds=30):
    """Extract audio, find loudest sample in first probe_seconds. Return
    time in seconds from video start."""
    wav = video_path + '.tmp.wav'
    run_q(['ffmpeg', '-y', '-i', video_path,
           '-t', str(probe_seconds),
           '-vn', '-ac', '1', '-ar', '16000', wav])
    import wave
    import struct
    with wave.open(wav, 'rb') as w:
        fs = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    os.remove(wav)
    samples = struct.unpack(f'<{n}h', raw)
    peak_idx = max(range(n), key=lambda i: abs(samples[i]))
    return peak_idx / fs


def audio_mode(videos):
    print('Detecting loudest transient in first 30s of each video...')
    peaks = {}
    for v in videos:
        cam = basename(v).split('.')[0]
        t = detect_audio_peak(v)
        peaks[cam] = t
        print(f'  {cam}: peak at t={t:.3f}s')
    ref = list(peaks.keys())[0]
    print(f'\nUsing {ref} as reference. Per-camera offsets (sec):')
    offsets = {}
    for cam, t in peaks.items():
        offset = t - peaks[ref]
        offsets[cam] = offset
        print(f'  {cam}: {offset:+.3f}s')
    return offsets


# ---------- Visual mode ----------

def extract_thumb_strip(video_path, probe_start, probe_duration,
                        thumb_every_sec, out_path):
    """Extract evenly-spaced thumbnails and tile horizontally."""
    fps_out = 1.0 / thumb_every_sec
    tmp_dir = out_path + '.frames'
    os.makedirs(tmp_dir, exist_ok=True)
    run_q(['ffmpeg', '-y',
           '-ss', str(probe_start),
           '-i', video_path,
           '-t', str(probe_duration),
           '-vf', f'fps={fps_out},scale=320:180',
           join(tmp_dir, '%04d.jpg')])
    # Tile horizontally with frame number labels
    import cv2
    import numpy as np
    files = sorted([join(tmp_dir, f) for f in os.listdir(tmp_dir)
                    if f.endswith('.jpg')])
    tiles = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        t_sec = probe_start + i * thumb_every_sec
        cv2.putText(img, f'{t_sec:.1f}s', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(img)
    if tiles:
        # Wrap into rows of 8
        rows = []
        for i in range(0, len(tiles), 8):
            r = np.hstack(tiles[i:i+8])
            if r.shape[1] < tiles[0].shape[1] * 8:
                pad = np.zeros((r.shape[0],
                                 tiles[0].shape[1]*8 - r.shape[1], 3),
                                 dtype=r.dtype)
                r = np.hstack([r, pad])
            rows.append(r)
        strip = np.vstack(rows)
        cv2.imwrite(out_path, strip)
    # Cleanup
    for f in files:
        os.remove(f)
    os.rmdir(tmp_dir)
    return out_path


def visual_mode(videos, probe_start, probe_duration, thumb_every_sec,
                out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for v in videos:
        cam = basename(v).split('.')[0]
        out = join(out_dir, f'{cam}_strip.jpg')
        print(f'[{cam}] extracting thumbs to {out}...')
        extract_thumb_strip(v, probe_start, probe_duration,
                             thumb_every_sec, out)
    print(f'\nDone. Open the strips in {out_dir} side-by-side, identify '
          f'a common event, note its timestamp in each, and fill in the '
          f'offsets manually.')
    print('\nThumbnails are labeled with seconds-from-probe-start.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['audio', 'visual'], required=True)
    ap.add_argument('--videos', nargs='+', required=True,
                    help='Paths to 8 camera videos')
    ap.add_argument('--probe_start', type=float, default=0.0,
                    help='visual mode: where to start probing (seconds)')
    ap.add_argument('--probe_duration', type=float, default=30.0,
                    help='visual mode: how many seconds to probe')
    ap.add_argument('--thumb_every_sec', type=float, default=0.5,
                    help='visual mode: thumbnail spacing')
    ap.add_argument('--out_dir', default='/tmp/sync_thumbs')
    ap.add_argument('--save_json', default=None,
                    help='audio mode: save resulting offsets to this JSON')
    args = ap.parse_args()

    if args.mode == 'audio':
        offsets = audio_mode(args.videos)
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump({'offsets_sec': offsets}, f, indent=2)
            print(f'\nsaved -> {args.save_json}')
    else:
        visual_mode(args.videos, args.probe_start, args.probe_duration,
                     args.thumb_every_sec, args.out_dir)


if __name__ == '__main__':
    main()
