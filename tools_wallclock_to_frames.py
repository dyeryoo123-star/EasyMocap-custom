"""Given a target wall-clock time and per-camera (start_time, actual_fps),
compute the frame index in each camera that corresponds to that time.

Example sync_meta.json:
{
  "D01": {"start": "13:02:18", "fps": 60.05},
  "D02": {"start": "13:02:17", "fps": 60.05},
  "D03": {"start": "13:02:16", "fps": 25.02},
  "D04": {"start": "13:02:15", "fps": 25.02},
  "D05": {"start": "13:02:18", "fps": 60.05},
  "D06": {"start": "13:02:18", "fps": 60.05},
  "D07": {"start": "13:02:17", "fps": 60.05},
  "D08": {"start": "13:02:17", "fps": 30.03}
}

Usage:
  python tools_wallclock_to_frames.py --meta sync_meta.json --target 13:03:00
  # → prints D01: 2522, D02: 2582, D03: 1101, ...
"""

import argparse
import json


def parse_time(s):
    """'13:03:00' or '13:03:00.500' → seconds since midnight (float)."""
    parts = s.split(':')
    h, m = int(parts[0]), int(parts[1])
    sec = float(parts[2])
    return h * 3600 + m * 60 + sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True,
                    help='JSON with per-cam {start, fps}')
    ap.add_argument('--target', required=True,
                    help='wall-clock time, e.g. 13:03:00 or 13:03:00.500')
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    t_target = parse_time(args.target)
    print(f'target wall time: {args.target} = {t_target}s')
    print(f'{"cam":<6}{"fps":>10}{"start":>12}{"Δt (s)":>10}{"frame":>10}')
    print('-' * 50)
    for cam in sorted(meta.keys()):
        info = meta[cam]
        t_start = parse_time(info['start'])
        dt = t_target - t_start
        fps = info['fps']
        frame = round(dt * fps)
        if frame < 0:
            note = '   (BEFORE start)'
        else:
            note = ''
        print(f'{cam:<6}{fps:>10.2f}{info["start"]:>12}{dt:>10.2f}'
              f'{frame:>10d}{note}')


if __name__ == '__main__':
    main()
