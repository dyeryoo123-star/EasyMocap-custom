"""Probe one or more videos and report:
  - declared fps (from container metadata)
  - actual fps (from frame count / duration)
  - resolution

Trusting only declared fps can be wrong (some cameras claim 30 but record 25,
or VFR-encoded files lie). Always verify against actual count.
"""

import argparse
import json
import os
import subprocess


def probe(path, count_frames=True):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if count_frames:
        cmd += ['-count_frames']
    cmd += ['-show_entries',
            'stream=r_frame_rate,duration,width,height,nb_read_frames,nb_frames',
            '-of', 'json', path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(r.stdout)['streams'][0]
    num, den = info['r_frame_rate'].split('/')
    declared_fps = float(num) / float(den)
    duration = float(info['duration'])
    nb = info.get('nb_read_frames') or info.get('nb_frames') or '0'
    actual_frames = int(nb)
    actual_fps = actual_frames / duration if duration > 0 else 0
    return {
        'path': path,
        'width': info['width'],
        'height': info['height'],
        'declared_fps': declared_fps,
        'actual_fps': actual_fps,
        'actual_frames': actual_frames,
        'duration_sec': duration,
        'mismatch': abs(declared_fps - actual_fps) > 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='+')
    ap.add_argument('--no_count', action='store_true',
                    help='skip slow frame-count (use only metadata)')
    args = ap.parse_args()

    print(f'{"file":<60}{"WxH":>12}{"declared":>10}{"actual":>10}  status')
    print('-' * 105)
    for v in args.videos:
        r = probe(v, count_frames=not args.no_count)
        name = os.path.basename(v)[:55]
        wxh = f'{r["width"]}x{r["height"]}'
        decl = f'{r["declared_fps"]:.2f}'
        act = f'{r["actual_fps"]:.2f}' if not args.no_count else 'N/A'
        status = '⚠ MISMATCH' if r['mismatch'] else '✓ ok'
        print(f'{name:<60}{wxh:>12}{decl:>10}{act:>10}  {status}')


if __name__ == '__main__':
    main()
