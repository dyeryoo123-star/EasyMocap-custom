"""Visualize the correspondence between 2D detections (red) and reprojected
3D keypoints (green) on each camera, plus error lines connecting them.

For each frame, builds an 8-cam mosaic.
"""

import argparse
import json
import os
import pickle

import cv2
import numpy as np

BODY25_SKEL = [
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
]


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real())) for i in range(nn.size())]
    return {c: {"K": fs.getNode(f"K_{c}").mat(),
                "dist": fs.getNode(f"dist_{c}").mat()} for c in names}


def read_extri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real())) for i in range(nn.size())]
    return {c: {"R": fs.getNode(f"Rot_{c}").mat(),
                "T": fs.getNode(f"T_{c}").mat()} for c in names}


def make_panel(cam, im, kp2d, kp3d_proj, tile_w=480, tile_h=270, flip180=True):
    """Draw one camera's panel with 2D (red) and 3D-reproj (green) overlay.
    """
    H, W = im.shape[:2]
    out = im.copy()
    # connections (using 3D reproj for skeleton lines, slightly transparent)
    for (a, b) in BODY25_SKEL:
        if a < len(kp3d_proj) and b < len(kp3d_proj):
            p1 = tuple(np.round(kp3d_proj[a]).astype(int))
            p2 = tuple(np.round(kp3d_proj[b]).astype(int))
            if all(-100 < v < W + 100 for v in (p1[0], p2[0])) and \
               all(-100 < v < H + 100 for v in (p1[1], p2[1])):
                cv2.line(out, p1, p2, (50, 200, 50), 3)
    # error lines (blue)
    for i in range(len(kp2d)):
        if i >= len(kp3d_proj): break
        x1, y1, c = kp2d[i]
        if c < 0.3: continue
        x2, y2 = kp3d_proj[i]
        if not (-100 < x2 < W + 100 and -100 < y2 < H + 100): continue
        err = np.hypot(x1 - x2, y1 - y2)
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                 (255, 100, 0), 2)  # blue line (BGR)
    # 3D reprojected points (green) - drawn first so 2D red sits on top
    for i, (x, y) in enumerate(kp3d_proj):
        if not (-100 < x < W + 100 and -100 < y < H + 100): continue
        cv2.circle(out, (int(x), int(y)), 7, (50, 200, 50), -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 1)
    # 2D detected points (red)
    for i, (x, y, c) in enumerate(kp2d):
        if c < 0.3: continue
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 230), -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 1)
    # mean reproj error label
    valid = kp2d[:, 2] > 0.3
    if valid.any() and len(kp3d_proj) > 0:
        n = min(len(kp2d), len(kp3d_proj))
        errs = np.linalg.norm(kp2d[:n, :2] - kp3d_proj[:n], axis=1)[valid[:n]]
        err_str = f'mean={errs.mean():.1f}px  med={np.median(errs):.1f}px'
    else:
        err_str = 'no overlap'
    # flip if needed
    if flip180:
        out = cv2.flip(out, -1)
    # add cam label + error
    cv2.rectangle(out, (0, 0), (W, 60), (40, 40, 70), -1)
    cv2.putText(out, cam[:3], (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3)
    cv2.putText(out, err_str, (130, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (200, 255, 200), 2)
    out = cv2.resize(out, (tile_w, tile_h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--output_root', required=True,
                    help='e.g. .../output_masked_v4')
    ap.add_argument('--kp2d_dir', default='keypoints2d_masked_v4',
                    help='subdir under output_root for 2D pkl files')
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    intri = read_intri(os.path.join(args.data_root, 'intri.yml'))
    extri = read_extri(os.path.join(args.data_root, 'extri.yml'))
    cams = sorted(intri.keys())
    fr = args.frame

    # Load 3D keypoints
    d = json.load(open(os.path.join(args.output_root, 'keypoints3d',
                                     f'{fr:06d}.json')))
    if isinstance(d, list):
        d = d[0]
    kp3d = np.array(d['keypoints3d'], dtype=np.float32)[:, :3]

    tiles = []
    for cam in cams:
        img_p = os.path.join(args.data_root, 'images', cam, f'{fr:06d}.jpg')
        im = cv2.imread(img_p)
        if im is None:
            tiles.append(np.zeros((270, 480, 3), np.uint8)); continue
        # Project 3D to this cam
        K = intri[cam]['K']; D = intri[cam]['dist']
        R = extri[cam]['R']; T = extri[cam]['T'].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(kp3d, rvec, T, K, D)
        proj = proj.reshape(-1, 2)
        # Load 2D ViTPose
        pkl_p = os.path.join(args.output_root, args.kp2d_dir, cam,
                              f'{fr:06d}.pkl')
        kp2d = pickle.load(open(pkl_p, 'rb'))['keypoints'][0]
        tile = make_panel(cam, im, kp2d, proj)
        tiles.append(tile)

    # 2x4 mosaic
    row1 = np.hstack(tiles[:4])
    row2 = np.hstack(tiles[4:])
    big = np.vstack([row1, row2])
    # legend bar
    bar_h = 60
    bar = np.full((bar_h, big.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, '2D (ViTPose) = red    3D reproj = green    error line = blue',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    final = np.vstack([bar, big])
    cv2.imwrite(args.out, final, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'wrote {args.out}, shape={final.shape}')


if __name__ == '__main__':
    main()
