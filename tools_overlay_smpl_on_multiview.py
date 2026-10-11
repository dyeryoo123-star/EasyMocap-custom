"""Overlay fitted SMPL mesh onto multi-view image frames.

This script renders a lightweight projected mesh directly with OpenCV. It is
intended for calibration/fit diagnosis: the original camera image is preserved
and the SMPL surface is projected using each camera's intrinsics/extrinsics.
"""

import argparse
import json
import os
import subprocess
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

from easymocap.mytools.camera_utils import read_camera
from tools_render_smpl_mesh import load_smpl_model, smpl_forward


def load_smpl_params(path, shape_mode='fitted', shape_scale=1.0):
    data = json.load(open(path))
    if isinstance(data, dict) and 'annots' in data:
        data = data['annots'][0]
    elif isinstance(data, list):
        data = data[0]
    betas = np.asarray(data['shapes'], dtype=np.float64).reshape(-1)
    if shape_mode == 'zero':
        betas = np.zeros_like(betas)
    elif shape_mode == 'scale':
        betas *= shape_scale
    poses = np.asarray(data['poses'], dtype=np.float64).reshape(-1)
    if poses.shape[0] == 69:
        poses = np.concatenate([np.zeros(3), poses])
    rh = np.asarray(data['Rh'], dtype=np.float64).reshape(-1)
    th = np.asarray(data['Th'], dtype=np.float64).reshape(-1)
    return betas, poses, rh, th


def project_vertices(vertices, cam, scale):
    k = cam['K'].copy().astype(np.float64)
    k[0] *= scale
    k[1] *= scale
    rvec = cam['Rvec'].astype(np.float64)
    tvec = cam['T'].astype(np.float64)
    dist = cam['dist'].astype(np.float64) if cam.get('dist') is not None else None
    pts2d, _ = cv2.projectPoints(vertices.astype(np.float64), rvec, tvec, k, dist)
    pts2d = pts2d[:, 0, :]
    xyz_cam = (cam['R'].astype(np.float64) @ vertices.T + tvec).T
    return pts2d, xyz_cam[:, 2]


def draw_mesh_overlay(image, vertices, faces, cam, alpha=0.35, scale=0.333,
                      face_stride=1, wire_stride=3):
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    pts2d, depth = project_vertices(vertices, cam, scale)

    valid = depth > 0.05
    face_depth = depth[faces].mean(axis=1)
    # Back-to-front painting gives a simple painter's-algorithm approximation.
    order = np.argsort(face_depth)[::-1]
    overlay = image.copy()
    fill_color = np.array([205, 185, 255], dtype=np.uint8)  # BGR lavender
    edge_color = (80, 55, 150)

    for idx in order[::face_stride]:
        tri_idx = faces[idx]
        if not valid[tri_idx].all():
            continue
        tri = pts2d[tri_idx]
        x0, y0 = tri.min(axis=0)
        x1, y1 = tri.max(axis=0)
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        poly = np.round(tri).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, fill_color.tolist(), lineType=cv2.LINE_AA)

    image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    # A sparse wireframe keeps the surface readable without turning the image
    # into a dense triangle net.
    for idx in order[::wire_stride]:
        tri_idx = faces[idx]
        if not valid[tri_idx].all():
            continue
        tri = np.round(pts2d[tri_idx]).astype(np.int32)
        x0, y0 = tri.min(axis=0)
        x1, y1 = tri.max(axis=0)
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        cv2.polylines(image, [tri.reshape(-1, 1, 2)], True, edge_color, 1, cv2.LINE_AA)

    return image


def make_contact_sheet(images, cols=4):
    rows = int(np.ceil(len(images) / cols))
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    canvas = np.full((rows * h, cols * w, 3), 245, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y, x = r * h, c * w
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', required=True)
    parser.add_argument('--camera_dir', required=True)
    parser.add_argument('--smpl_dir', required=True)
    parser.add_argument('--smpl_model', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--mp4', required=True)
    parser.add_argument('--cams', nargs='+', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--panel_width', type=int, default=640)
    parser.add_argument('--alpha', type=float, default=0.35)
    parser.add_argument('--shape_mode', default='zero', choices=['fitted', 'zero', 'scale'])
    parser.add_argument('--shape_scale', type=float, default=1.0)
    parser.add_argument('--face_stride', type=int, default=1)
    parser.add_argument('--wire_stride', type=int, default=5)
    args = parser.parse_args()

    cams = read_camera(join(args.camera_dir, 'intri.yml'), join(args.camera_dir, 'extri.yml'))
    smpl = load_smpl_model(args.smpl_model)
    faces = np.asarray(smpl['f'], dtype=np.int32)
    smpl_files = sorted(glob(join(args.smpl_dir, '*.json')))
    if args.end > 0:
        smpl_files = smpl_files[args.start:args.end]
    else:
        smpl_files = smpl_files[args.start:]
    if not smpl_files:
        raise RuntimeError(f'No SMPL files found in {args.smpl_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    for local_idx, smpl_path in enumerate(smpl_files):
        frame_name = basename(smpl_path).replace('.json', '.jpg')
        betas, poses, rh, th = load_smpl_params(smpl_path, args.shape_mode, args.shape_scale)
        vertices, _ = smpl_forward(smpl, betas, poses, rh, th)
        panels = []
        for cam_name in args.cams:
            img_path = join(args.images_root, cam_name, frame_name)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(img_path)
            scale = args.panel_width / img.shape[1]
            panel = draw_mesh_overlay(
                img, vertices, faces, cams[cam_name], alpha=args.alpha,
                scale=scale, face_stride=args.face_stride,
                wire_stride=args.wire_stride,
            )
            cv2.putText(panel, cam_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(panel, cam_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (20, 20, 20), 1, cv2.LINE_AA)
            panels.append(panel)
        sheet = make_contact_sheet(panels, cols=4)
        out_path = join(args.out_dir, f'{local_idx:06d}.jpg')
        cv2.imwrite(out_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if (local_idx + 1) % 10 == 0 or local_idx == len(smpl_files) - 1:
            print(f'[overlay] rendered {local_idx + 1}/{len(smpl_files)}')

    os.makedirs(os.path.dirname(args.mp4), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(args.fps),
        '-i', join(args.out_dir, '%06d.jpg'),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        args.mp4,
    ]
    subprocess.run(cmd, check=True)
    print(f'[overlay] wrote {args.mp4}')


if __name__ == '__main__':
    main()
