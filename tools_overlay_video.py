"""Render SMPL mesh overlay onto camera images across a frame range and
compose into MP4. Lets us visually inspect whether the fitted 3D body
tracks the actual person throughout an action sequence.

Usage:
  python tools_overlay_video.py --cam D02_20260401170555 --smpl_dir <smpl/> \
      --image_root <images/> --frame_start 320 --frame_end 360 \
      --intri intri.yml --extri extri.yml --mp4 /tmp/overlay.mp4
"""

import argparse
import json
import os
import subprocess
from glob import glob
from os.path import join

import cv2
import numpy as np
import pyrender
import trimesh
from easymocap.smplmodel import load_model


def look_at_pose(eye, target, up=np.array([0.0, 1.0, 0.0])):
    z = eye - target
    z = z / max(np.linalg.norm(z), 1e-8)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 0.01:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', required=True)
    ap.add_argument('--smpl_dir', required=True)
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--smpl_frame_start', type=int, default=320,
                    help='SMPL JSON filename starting index')
    ap.add_argument('--frame_count', type=int, default=40)
    ap.add_argument('--image_file_pattern', default='{file_index:06d}.jpg',
                    help='how SMPL frame N maps to image filename')
    ap.add_argument('--image_index_offset', type=int, default=10*320 - 320,
                    help='offset added to SMPL index to get image file number')
    ap.add_argument('--intri', required=True)
    ap.add_argument('--extri', required=True)
    ap.add_argument('--out_frames', default='/tmp/overlay_frames')
    ap.add_argument('--mp4', required=True)
    ap.add_argument('--fps', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_frames, exist_ok=True)

    # Cameras
    fs_i = cv2.FileStorage(args.intri, cv2.FILE_STORAGE_READ)
    K = fs_i.getNode(f'K_{args.cam}').mat()
    dist = fs_i.getNode(f'dist_{args.cam}').mat().reshape(-1)
    fs_e = cv2.FileStorage(args.extri, cv2.FILE_STORAGE_READ)
    R = fs_e.getNode(f'Rot_{args.cam}').mat()
    T = fs_e.getNode(f'T_{args.cam}').mat().reshape(3)

    # Pyrender camera pose
    cam_T_world = np.eye(4); cam_T_world[:3, :3] = R; cam_T_world[:3, 3] = T
    world_T_cam = np.linalg.inv(cam_T_world)
    flip_yz = np.diag([1, -1, -1, 1]).astype(np.float64)
    world_T_cam_gl = world_T_cam @ flip_yz

    # SMPL body
    body = load_model(gender='neutral', model_type='smpl')
    faces = body.faces

    # Probe one image for size
    sample = sorted(glob(join(args.image_root, args.cam, '*.jpg')))[0]
    img0 = cv2.imread(sample)
    H, W = img0.shape[:2]

    # Persistent renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    skipped = 0
    for i in range(args.frame_count):
        smpl_idx = args.smpl_frame_start + i
        # Map SMPL JSON index → actual image filename. In v8 the SMPL JSON
        # files are named after dataset index; image files are at every-10
        # frame numbering, so image filename = smpl_idx * 10.
        img_idx = smpl_idx * 10
        smpl_path = join(args.smpl_dir, f'{smpl_idx:06d}.json')
        img_path = join(args.image_root, args.cam, f'{img_idx:06d}.jpg')
        if not os.path.exists(smpl_path) or not os.path.exists(img_path):
            skipped += 1
            continue

        # SMPL params
        d = json.load(open(smpl_path))
        if not d:
            skipped += 1; continue
        d0 = d[0]
        params = {k: np.array(d0[k], dtype=np.float32)
                   for k in ['Rh', 'Th', 'poses', 'shapes']}
        verts = body(return_verts=True, return_tensor=False, **params)[0]

        # Image
        img = cv2.imread(img_path)
        img_undist = cv2.undistort(img, K, dist)

        # Scene
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.78, 0.76, 0.92, 0.95],
            metallicFactor=0.0, roughnessFactor=0.4, alphaMode='BLEND')
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0],
                                 ambient_light=[0.55, 0.55, 0.62])
        scene.add(pyrender.Mesh.from_trimesh(mesh_tm, material=mat,
                                              smooth=True))
        intri_cam = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            znear=0.05, zfar=100.0)
        scene.add(intri_cam, pose=world_T_cam_gl)

        center = verts.mean(axis=0)
        for offset in [(2, 2, 2), (-2, 2, 2), (0, -2, 2), (0, 0, -3)]:
            light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
            eye = center + np.array(offset)
            scene.add(light, pose=look_at_pose(eye, center))

        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        # Blend
        alpha = (color[..., 3:4].astype(np.float32) / 255.0) * 0.92
        mesh_rgb = color[..., :3].astype(np.float32)
        bg = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB).astype(np.float32)
        blended = (mesh_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        out_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # Outline
        mask = (color[..., 3] > 30).astype(np.uint8) * 255
        edges = cv2.Canny(mask, 50, 150)
        edges_dil = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        out_bgr[edges_dil > 0] = (40, 40, 60)

        # Label
        cv2.rectangle(out_bgr, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.putText(out_bgr, f'{args.cam[:3]}  frame {img_idx:06d}  '
                     f'(SMPL idx {smpl_idx})',
                     (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                     (255, 255, 255), 2)
        cv2.imwrite(join(args.out_frames, f'{i:06d}.png'), out_bgr)

    renderer.delete()
    print(f'rendered {args.frame_count - skipped}/{args.frame_count} frames '
          f'(skipped {skipped})')

    # Compose
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps),
           '-i', join(args.out_frames, '%06d.png'),
           '-vf', f'scale={W}:{H},pad=ceil(iw/2)*2:ceil(ih/2)*2',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', args.mp4]
    subprocess.run(cmd, check=True)
    print(f'mp4 -> {args.mp4}')


if __name__ == '__main__':
    main()
