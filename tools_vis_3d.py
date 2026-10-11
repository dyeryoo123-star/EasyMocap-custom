"""Render keypoints3d as a rotating 3D skeleton video.

Reads a directory of <frame>.json files (output of mv1p.py's --skel step)
and produces a frame-by-frame matplotlib 3D animation, then composes into MP4.

Usage:
  python tools_vis_3d.py \
      --kp3d_dir calibration_data/mocap_10000_10700/output_ba_extri_vitpose/keypoints3d \
      --out_dir /tmp/vis3d_frames --mp4 /tmp/vis3d.mp4 --fps 5
"""

import argparse
import json
import os
import subprocess
from glob import glob
from os.path import join

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# OpenPose body25 skeleton (parent, child) pairs
BODY25_EDGES = [
    (1, 0), (1, 2), (1, 5), (1, 8),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (8, 9), (8, 12),
    (9, 10), (10, 11),
    (12, 13), (13, 14),
    (0, 15), (0, 16), (15, 17), (16, 18),
    (11, 22), (11, 24), (22, 23),
    (14, 19), (14, 21), (19, 20),
]


def load_kp3d(path):
    d = json.load(open(path))
    if not d:
        return None
    k = np.array(d[0]['keypoints3d'], dtype=np.float32)
    return k  # (25, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kp3d_dir', required=True)
    ap.add_argument('--out_dir', required=True,
                    help='where per-frame PNGs go')
    ap.add_argument('--mp4', required=True, help='final mp4 output')
    ap.add_argument('--fps', type=int, default=5)
    ap.add_argument('--conf_min', type=float, default=0.2)
    ap.add_argument('--azim_speed', type=float, default=2.0,
                    help='degrees per frame the view azim rotates')
    ap.add_argument('--elev', type=float, default=20.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(join(args.kp3d_dir, '*.json')))
    print(f'[vis] found {len(files)} frames')

    # Load all to compute bounding box (so axes stay stable)
    all_pts = []
    for f in files:
        k = load_kp3d(f)
        if k is None:
            continue
        vis = k[:, 3] > args.conf_min
        if vis.any():
            all_pts.append(k[vis, :3])
    if not all_pts:
        print('[vis] no valid frames'); return
    all_pts = np.concatenate(all_pts, axis=0)
    center = all_pts.mean(axis=0)
    radius = max(1.2, float(np.linalg.norm(all_pts - center, axis=1).max()) * 1.05)
    lo, hi = center - radius, center + radius

    for i, f in enumerate(files):
        k = load_kp3d(f)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        if k is not None:
            vis = k[:, 3] > args.conf_min
            xyz = k[:, :3]
            # joints
            ax.scatter(xyz[vis, 0], xyz[vis, 1], xyz[vis, 2],
                        c='red', s=30, depthshade=True)
            # bones
            for (a, b) in BODY25_EDGES:
                if a < len(k) and b < len(k) and vis[a] and vis[b]:
                    ax.plot([xyz[a, 0], xyz[b, 0]],
                             [xyz[a, 1], xyz[b, 1]],
                             [xyz[a, 2], xyz[b, 2]],
                             c='blue', linewidth=2)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(f'Frame {i:03d}  ({os.path.basename(f)})')
        # Rotating view
        ax.view_init(elev=args.elev, azim=(i * args.azim_speed) % 360)
        out_png = join(args.out_dir, f'{i:06d}.png')
        plt.savefig(out_png, dpi=80, bbox_inches='tight')
        plt.close(fig)
        if (i + 1) % 10 == 0:
            print(f'[vis] rendered {i+1}/{len(files)}')

    print(f'[vis] composing mp4: {args.mp4}')
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps),
           '-i', join(args.out_dir, '%06d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.mp4]
    subprocess.run(cmd, check=True)
    print(f'[vis] done -> {args.mp4}')


if __name__ == '__main__':
    main()
