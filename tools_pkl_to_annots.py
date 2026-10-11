"""Convert myeasymocap ViTPose pkl caches into easymocap annots json format.

ViTPose pkl per (cam, frame):
    {'keypoints': ndarray of shape (1, 25, 3) for single_person}

Target annots json per (cam, frame):
    {'filename': '...', 'height': H, 'width': W,
     'annots': [{'personID': 0, 'bbox': [x1,y1,x2,y2,conf],
                 'keypoints': [[x, y, conf], ...]}]}

Reads the YOLO 'person/' npy files for bbox (when available).
"""

import argparse
import json
import os
import pickle
from glob import glob
from os.path import join

import cv2
import numpy as np

from myeasymocap.backbone.vitpose.vit_plain import coco25tobody25


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True,
                    help='directory with keypoints2d/<cam>/<frame>.pkl '
                         '(and person/<cam>/<frame>.npy)')
    ap.add_argument('--dst_root', required=True,
                    help='dataset root; new annots written to '
                         '<dst_root>/<annot_dir>/<cam>/<frame>.json')
    ap.add_argument('--annot_dir', default='annots_vitpose')
    ap.add_argument('--image_dir', default='images',
                    help='for reading image shape')
    ap.add_argument('--fallback_zero', action='store_true',
                    help='for frames missing pkl, write empty annot json '
                         '(needed when downstream expects all frames present)')
    args = ap.parse_args()

    kp2d_root = join(args.src, 'keypoints2d')
    person_root = join(args.src, 'person')
    out_root = join(args.dst_root, args.annot_dir)
    img_root = join(args.dst_root, args.image_dir)

    cams = sorted([c for c in os.listdir(kp2d_root)
                    if os.path.isdir(join(kp2d_root, c))])
    print(f'[convert] {len(cams)} cams: {cams}')

    # Cache image dims per cam by reading first image
    img_dims = {}
    for cam in cams:
        files = sorted(glob(join(img_root, cam, '*.jpg')))
        if files:
            img = cv2.imread(files[0])
            img_dims[cam] = (img.shape[0], img.shape[1])

    n_total_out = 0
    n_zero_fill = 0
    for cam in cams:
        os.makedirs(join(out_root, cam), exist_ok=True)
        H, W = img_dims.get(cam, (1080, 1920))
        # Frames from images dir (canonical full list)
        img_files = sorted(glob(join(img_root, cam, '*.jpg')))
        all_frames = [os.path.splitext(os.path.basename(p))[0]
                      for p in img_files]
        for frame in all_frames:
            pkl_path = join(kp2d_root, cam, f'{frame}.pkl')
            npy_path = join(person_root, cam, f'{frame}.npy')
            kpts = None
            bbox = [0, 0, 100, 100, 0]  # placeholder
            if os.path.exists(pkl_path):
                d = load_pkl(pkl_path)
                kp = d['keypoints']
                if kp.ndim == 3 and kp.shape[0] >= 1:
                    kpts = kp[0]  # (25, 3) — still in coco_25 order
                elif kp.ndim == 2:
                    kpts = kp
                # Convert coco_25 ordering -> OpenPose body25
                kpts = coco25tobody25(kpts[None])[0]
                if os.path.exists(npy_path):
                    arr = np.load(npy_path, allow_pickle=True)
                    # Find first 'person' bbox in YOLO results
                    if hasattr(arr, 'tolist'):
                        try:
                            results = arr.item().get('results', [])
                        except Exception:
                            results = []
                        for r in results:
                            if len(r) >= 7 and r[6] == 'person':
                                bbox = [float(r[0]), float(r[1]),
                                        float(r[2]), float(r[3]),
                                        float(r[4])]
                                break
            if kpts is None:
                if not args.fallback_zero:
                    continue
                kpts = np.zeros((25, 3), dtype=np.float32)
                n_zero_fill += 1
            out = {
                'filename': join(args.image_dir, cam, f'{frame}.jpg'),
                'height': int(H), 'width': int(W),
                'annots': [{
                    'personID': 0,
                    'bbox': bbox,
                    'keypoints': kpts.tolist(),
                    'isKeyframe': False,
                }],
            }
            with open(join(out_root, cam, f'{frame}.json'), 'w') as f:
                json.dump(out, f, indent=2)
            n_total_out += 1
    print(f'[convert] wrote {n_total_out} json files (zero-filled: {n_zero_fill})')
    print(f'[convert] output: {out_root}')


if __name__ == '__main__':
    main()
