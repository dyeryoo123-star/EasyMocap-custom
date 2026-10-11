"""Render SMPL mesh animation from per-frame SMPL params (mv1p.py output).

Reads <root>/output_*/smpl/<frame>.json files, computes SMPL vertices via the
neutral body model, renders the mesh from a fixed (or slowly rotating) camera
using pyrender (offscreen). Composes frames into MP4.

Usage:
  python tools_vis_smpl_mesh.py \
      --smpl_dir calibration_data/mocap_10000_10700/output_ba_extri_vitpose/smpl \
      --out_dir /tmp/vis_smpl_frames --mp4 /tmp/vis_smpl_mesh.mp4 --fps 5
"""

import os
# Note: on macOS we rely on native OpenGL via Quartz; don't set PYOPENGL_PLATFORM.
# On Linux servers without display, set PYOPENGL_PLATFORM=osmesa or egl
# externally before running this script.

import argparse
import json
import subprocess
from glob import glob
from os.path import join

import numpy as np
import trimesh
import pyrender
from easymocap.smplmodel import load_model


def load_smpl_params(path):
    d = json.load(open(path))
    if not d:
        return None
    p = d[0]
    return {
        'Rh': np.array(p['Rh'], dtype=np.float32),
        'Th': np.array(p['Th'], dtype=np.float32),
        'poses': np.array(p['poses'], dtype=np.float32),
        'shapes': np.array(p['shapes'], dtype=np.float32),
    }


def setup_scene_static(verts_all, view_dir, look_at_center):
    """Compute a scene-stable camera placement enclosing all frames."""
    pts = np.concatenate(verts_all, axis=0)
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).max()) * 1.3
    return center, radius


def look_at_pose(eye, target, up=np.array([0., -1., 0.])):
    """Build a camera-to-world transform looking from eye to target."""
    fwd = (target - eye)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    # In pyrender / OpenGL, camera looks down -Z; we'll build accordingly
    z = -fwd
    x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smpl_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mp4', required=True)
    ap.add_argument('--fps', type=int, default=5)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--azim_speed', type=float, default=3.0,
                    help='degrees per frame for camera orbit')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob(join(args.smpl_dir, '*.json')))
    if args.end > 0:
        files = files[args.start:args.end]
    else:
        files = files[args.start:]
    print(f'[smpl] {len(files)} frames')

    print('[smpl] loading body model...')
    body = load_model(gender='neutral', model_type='smpl')

    # First pass: compute all vertices to find a good scene bbox
    all_verts = []
    cached = {}
    for f in files:
        p = load_smpl_params(f)
        if p is None:
            continue
        v = body(return_verts=True, return_tensor=False, **p)
        v = np.asarray(v)[0]  # (6890, 3)
        cached[f] = v
        all_verts.append(v)
    if not all_verts:
        print('[smpl] no valid frames'); return
    center, radius = setup_scene_static(all_verts, None, None)
    print(f'[smpl] scene center={center}, radius={radius:.2f}m')

    # Renderer
    r = pyrender.OffscreenRenderer(viewport_width=args.width,
                                    viewport_height=args.height)
    faces = body.faces

    for i, f in enumerate(files):
        if f not in cached:
            # Skip — render an empty frame to keep timing
            scene = pyrender.Scene(bg_color=[0.15, 0.15, 0.18, 1.0])
        else:
            verts = cached[f]
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            scene = pyrender.Scene(bg_color=[0.15, 0.15, 0.18, 1.0],
                                    ambient_light=[0.35, 0.35, 0.35])
            skin_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.78, 0.70, 0.59, 1.0],
                metallicFactor=0.0, roughnessFactor=0.7)
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=skin_mat,
                                                   smooth=True))
            # Ground plane: feet (small Y because person stands on ground in
            # world frame where Y is up). Place ground just below the
            # smallest Y observed across all frames.
            gy = float(np.array(all_verts).reshape(-1, 3)[:, 1].min() - 0.05)
            ground = trimesh.creation.box(extents=[radius * 3, 0.005, radius * 3])
            T = np.eye(4); T[:3, 3] = [center[0], gy, center[2]]
            ground_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.30, 0.30, 0.30, 1.0],
                metallicFactor=0.0, roughnessFactor=1.0)
            scene.add(pyrender.Mesh.from_trimesh(ground, material=ground_mat,
                                                   smooth=False), pose=T)

        # Camera: orbiting around the scene center
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        azim = np.deg2rad((i * args.azim_speed) % 360)
        elev = np.deg2rad(15.0)
        eye = center + radius * np.array([
            np.cos(elev) * np.sin(azim),
            np.sin(elev),  # camera slightly above scene center (Y is "up")
            np.cos(elev) * np.cos(azim),
        ])
        cam_pose = look_at_pose(eye, center, up=np.array([0., 1., 0.]))
        scene.add(cam, pose=cam_pose)
        # Two directional lights for nicer shading
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        light_pose = look_at_pose(eye + np.array([0, -1, 0]), center,
                                   up=np.array([0., 0., 1.]))
        scene.add(light, pose=light_pose)

        color, _ = r.render(scene)
        out_png = join(args.out_dir, f'{i:06d}.png')
        from PIL import Image
        Image.fromarray(color).save(out_png)
        if (i + 1) % 10 == 0:
            print(f'[smpl] rendered {i+1}/{len(files)}')

    r.delete()

    print(f'[smpl] composing mp4: {args.mp4}')
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps),
           '-i', join(args.out_dir, '%06d.png'),
           '-vf', f'scale={args.width}:{args.height},pad=ceil(iw/2)*2:ceil(ih/2)*2',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.mp4]
    subprocess.run(cmd, check=True)
    print(f'[smpl] done -> {args.mp4}')


if __name__ == '__main__':
    main()
