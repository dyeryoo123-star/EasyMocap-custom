"""Render SMPL mesh from fitted parameters using Open3D offscreen renderer.

Produces a clean 3D mesh image like the user's reference SMPL Mesh viewer:
purplish skin tone, no background, soft lighting, isometric/front view.
"""

import argparse
import json
import os
import pickle

import cv2
import numpy as np
import open3d as o3d
import torch


def load_smpl_model(model_path):
    """Load SMPL model from .pkl. Returns shape regressor + pose blendshapes."""
    with open(model_path, 'rb') as f:
        sm = pickle.load(f, encoding='latin1')
    return sm


def smpl_forward(smpl, betas, poses, Rh, Th):
    """Minimal SMPL forward pass: returns vertices in world space.

    betas:  (10,)
    poses:  (72,) - axis-angle for 24 joints (3 each)
    Rh:     (3,)  - global orientation axis-angle
    Th:     (3,)  - global translation
    """
    # Chumpy-aware fetching (SMPL pkls bundle some arrays as chumpy 'Select'
    # objects). Use the `.r` attribute when available.
    def asnp(x):
        if x is None:
            return None
        if hasattr(x, 'r'):
            return np.asarray(x.r)
        if hasattr(x, 'toarray'):
            return x.toarray()
        return np.asarray(x)

    v_template  = asnp(smpl['v_template']).astype(np.float64)
    shapedirs   = asnp(smpl['shapedirs'])[:, :, :10].astype(np.float64)
    posedirs    = asnp(smpl['posedirs']).astype(np.float64)
    J_regressor = asnp(smpl['J_regressor']).astype(np.float64)
    weights     = asnp(smpl['weights']).astype(np.float64)
    kintree_table = asnp(smpl['kintree_table'])
    parent = kintree_table[0].astype(np.int32)

    # 1. shape blend
    v_shaped = v_template + np.einsum('vci,i->vc', shapedirs, betas)
    # 2. joint locations
    J = J_regressor @ v_shaped  # (24, 3)
    # 3. pose blend shapes
    full_pose = poses.reshape(24, 3)
    # rodrigues for each joint
    Rs = np.stack([rodrigues(full_pose[i]) for i in range(24)], axis=0)
    # subtract identity (root rotation excluded from pose blendshapes)
    R_minus_I = (Rs[1:] - np.eye(3)).reshape(-1)  # (23*9 = 207,)
    v_posed = v_shaped + np.einsum('vci,i->vc', posedirs, R_minus_I)
    # 4. forward kinematics
    G = np.zeros((24, 4, 4))
    G[0] = np.eye(4); G[0, :3, :3] = Rs[0]; G[0, :3, 3] = J[0]
    for i in range(1, 24):
        T_local = np.eye(4)
        T_local[:3, :3] = Rs[i]
        T_local[:3, 3] = J[i] - J[parent[i]]
        G[i] = G[parent[i]] @ T_local
    # remove rest pose offset
    G_canonical = G.copy()
    for i in range(24):
        G_canonical[i, :3, 3] -= G[i, :3, :3] @ J[i]
    # 5. skinning
    T = np.einsum('vk,kij->vij', weights, G_canonical)
    v_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=-1)
    v_world = np.einsum('vij,vj->vi', T, v_h)[:, :3]
    # 6. EasyMocap stores root/global rotation separately as Rh. The pose
    # vector keeps the SMPL root joint neutral, then Rh/Th move the whole body
    # in the capture world.
    R_global = rodrigues(Rh)
    v_world = v_world @ R_global.T + Th
    f_np = asnp(smpl['f']).astype(np.int32)
    return v_world.astype(np.float32), f_np


def rodrigues(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-9:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def render_mesh(verts, faces, out_path, view='front', size=1280):
    """Render SMPL mesh using matplotlib's Poly3DCollection (cross-platform,
    no offscreen GL required).

    Produces a clean shaded mesh on white background, similar to the
    canonical SMPL viewer reference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Center mesh.
    center = (verts.max(0) + verts.min(0)) / 2
    v = verts - center

    # Coordinate remap so the body appears upright in matplotlib (matplotlib
    # uses Z as the vertical axis by default; our world frame has Y as the
    # gravity axis with head at +Y / feet at -Y).
    #   plot_X = world_X         (right-left)
    #   plot_Y = world_Z         (depth)
    #   plot_Z = world_Y         (vertical, +Y = up = head)
    v = np.stack([v[:, 0], v[:, 2], v[:, 1]], axis=-1)

    # camera setups: (azim, elev) for matplotlib's view_init
    views = {
        'front': (-90, 10),
        'side':  (  0, 10),
        'iso':   (-55, 20),
        'back':  ( 90, 10),
    }
    azim, elev = views.get(view, views['front'])

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100,
                     facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Build polygons with per-face shading based on a fixed light direction.
    tri = v[faces]                              # (F, 3, 3)
    centroids = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0],
                       tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    light_dir = np.array([0.4, -0.6, -1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shading = np.clip(normals @ light_dir, 0.0, 1.0)        # (F,)
    shading = 0.45 + 0.55 * shading
    base = np.array([0.78, 0.75, 0.92])                     # SMPL viewer purple
    face_colors = np.outer(shading, base)
    face_colors = np.clip(face_colors, 0, 1)
    face_colors = np.hstack([face_colors, np.ones((face_colors.shape[0], 1))])

    # Z-order: paint back-to-front by centroid distance from camera
    cam_dir = np.array([
        np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
        np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
        np.sin(np.radians(elev)),
    ])
    depth = centroids @ cam_dir
    order = np.argsort(-depth)
    poly = Poly3DCollection(tri[order], facecolors=face_colors[order],
                            edgecolors='none', linewidths=0,
                            antialiased=True)
    ax.add_collection3d(poly)

    # set limits
    R = float(np.abs(v).max()) * 1.1
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='white')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smpl_model', required=True)
    ap.add_argument('--smpl_json',  required=True,
                    help='path to one frame of SMPL output (e.g. 000600.json)')
    ap.add_argument('--out',        required=True)
    ap.add_argument('--view',       default='front',
                    choices=['front', 'side', 'iso', 'back'])
    args = ap.parse_args()

    sm = load_smpl_model(args.smpl_model)
    d = json.load(open(args.smpl_json))
    if isinstance(d, dict) and 'annots' in d:
        d = d['annots'][0]
    elif isinstance(d, list):
        d = d[0]
    betas = np.array(d['shapes']).flatten()
    poses = np.array(d['poses']).flatten()
    Rh    = np.array(d['Rh']).flatten()
    Th    = np.array(d['Th']).flatten()
    # Full 72-dim pose: first 3 are root rotation (we use Rh instead)
    if poses.shape[0] == 69:
        full_pose = np.concatenate([np.zeros(3), poses])
    else:
        full_pose = poses
    verts, faces = smpl_forward(sm, betas, full_pose, Rh, Th)
    print(f'verts: {verts.shape}, faces: {faces.shape}')
    print(f'verts bbox: min={verts.min(0)} max={verts.max(0)}')
    ok = render_mesh(verts, faces, args.out, view=args.view)
    if ok:
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
