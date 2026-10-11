"""Render a fixed-camera SMPL mesh sequence to frames and MP4.

This uses matplotlib instead of pyrender so it works in headless/macOS
sessions where OpenGL offscreen rendering may not have a display context.
"""

import argparse
import json
import os
import subprocess
from glob import glob
from os.path import join

import numpy as np

from tools_render_smpl_mesh import load_smpl_model, smpl_forward


def load_params(path, shape_mode='fitted', shape_scale=1.0):
    data = json.load(open(path))
    if isinstance(data, dict) and 'annots' in data:
        data = data['annots'][0]
    elif isinstance(data, list):
        data = data[0]
    betas = np.array(data['shapes'], dtype=np.float64).flatten()
    if shape_mode == 'zero':
        betas = np.zeros_like(betas)
    elif shape_mode == 'scale':
        betas = betas * shape_scale
    poses = np.array(data['poses'], dtype=np.float64).flatten()
    rh = np.array(data['Rh'], dtype=np.float64).flatten()
    th = np.array(data['Th'], dtype=np.float64).flatten()
    if poses.shape[0] == 69:
        poses = np.concatenate([np.zeros(3), poses])
    return betas, poses, rh, th


def remap_for_plot(verts):
    # Matplotlib uses Z as vertical. EasyMocap world here has Z as vertical.
    # Keep world X and Z visually intuitive, and put Y as depth.
    return np.stack([verts[:, 0], verts[:, 1], verts[:, 2]], axis=-1)


def read_camera_center_from_extri(extri_path, camera_name):
    import cv2

    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f'Cannot open extri file: {extri_path}')
    R = fs.getNode(f'Rot_{camera_name}').mat()
    T = fs.getNode(f'T_{camera_name}').mat()
    fs.release()
    if R is None or T is None:
        raise KeyError(f'Camera {camera_name} not found in {extri_path}')
    T = T.reshape(3, 1)
    return (-R.T @ T).reshape(3)


def camera_view_angles(center, camera_center):
    # Matplotlib's azimuth/elevation define the direction from the object
    # center toward the virtual camera in data coordinates.
    direction = remap_for_plot(camera_center[None])[0] - remap_for_plot(center[None])[0]
    xy = float(np.hypot(direction[0], direction[1]))
    azim = float(np.degrees(np.arctan2(direction[1], direction[0])))
    elev = float(np.degrees(np.arctan2(direction[2], xy)))
    return azim, elev


def render_frame(verts, faces, out_path, center, radius, view, size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    views = {
        'front': (-90, 8),
        'back': (90, 8),
        'side': (0, 8),
        'body_front': (180, 8),
        'iso': (-55, 18),
        'top': (-90, 70),
    }
    if isinstance(view, (tuple, list)):
        azim, elev = view
    else:
        azim, elev = views.get(view, views['iso'])

    v = remap_for_plot(verts)
    c = remap_for_plot(center[None])[0]
    tri = v[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    light_dir = np.array([0.35, -0.55, 0.95])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shade = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)
    base = np.array([0.74, 0.72, 0.92])
    colors = np.hstack([np.outer(shade, base), np.ones((shade.shape[0], 1))])

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    poly = Poly3DCollection(tri, facecolors=colors, edgecolors='none', linewidths=0)
    ax.add_collection3d(poly)
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_model', required=True)
    parser.add_argument('--smpl_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--mp4', required=True)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--view', default='iso', choices=['front', 'back', 'side', 'body_front', 'iso', 'top', 'camera'])
    parser.add_argument('--extri', default=None, help='EasyMocap extri.yml, required for --view camera')
    parser.add_argument('--camera', default=None, help='Camera name, e.g. D05_20260325, required for --view camera')
    parser.add_argument('--size', type=int, default=960)
    parser.add_argument('--shape_mode', default='fitted', choices=['fitted', 'zero', 'scale'])
    parser.add_argument('--shape_scale', type=float, default=1.0)
    args = parser.parse_args()

    files = sorted(glob(join(args.smpl_dir, '*.json')))
    if args.end > 0:
        files = files[args.start:args.end]
    else:
        files = files[args.start:]
    if not files:
        raise RuntimeError(f'No SMPL json files found in {args.smpl_dir}')

    smpl = load_smpl_model(args.smpl_model)
    vertices = []
    for path in files:
        betas, poses, rh, th = load_params(path, args.shape_mode, args.shape_scale)
        verts, faces = smpl_forward(smpl, betas, poses, rh, th)
        vertices.append(verts)

    all_points = np.concatenate(vertices, axis=0)
    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2.0
    radius = float(np.max(np.abs(all_points - center))) * 1.15
    print(f'[mesh] frames={len(files)}, center={center}, radius={radius:.3f}')
    view = args.view
    if args.view == 'camera':
        if not args.extri or not args.camera:
            raise ValueError('--view camera requires --extri and --camera')
        cam_center = read_camera_center_from_extri(args.extri, args.camera)
        view = camera_view_angles(center, cam_center)
        print(f'[mesh] camera={args.camera}, camera_center={cam_center}, azim={view[0]:.2f}, elev={view[1]:.2f}')

    os.makedirs(args.out_dir, exist_ok=True)
    for idx, verts in enumerate(vertices):
        out_path = join(args.out_dir, f'{idx:06d}.png')
        render_frame(verts, faces, out_path, center, radius, view, args.size)
        if (idx + 1) % 10 == 0 or idx == len(vertices) - 1:
            print(f'[mesh] rendered {idx + 1}/{len(vertices)}')

    os.makedirs(os.path.dirname(args.mp4), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(args.fps),
        '-i', join(args.out_dir, '%06d.png'),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        args.mp4,
    ]
    subprocess.run(cmd, check=True)
    print(f'[mesh] wrote {args.mp4}')


if __name__ == '__main__':
    main()
