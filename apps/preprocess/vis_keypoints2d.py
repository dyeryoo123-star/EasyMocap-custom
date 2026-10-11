import argparse
import os
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

from easymocap.mytools.file_utils import read_json


def draw_one(img, ann, conf_thres=0.2):
    bbox = ann.get("bbox", [0, 0, 0, 0, 0])
    if len(bbox) >= 4:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 40), 2)
    kpts = np.asarray(ann.get("keypoints", []), dtype=np.float32)
    if kpts.ndim == 2 and kpts.shape[1] >= 3:
        for i, (x, y, c) in enumerate(kpts):
            if c < conf_thres:
                continue
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (30, 30, 255), -1)
            cv2.putText(img, str(i), (int(round(x)) + 2, int(round(y)) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--annot", type=str, default="annots_upright_v1_clean")
    parser.add_argument("--out", type=str, default="vis2d_upright_v1_clean")
    parser.add_argument("--subs", nargs="+", default=[])
    parser.add_argument("--ext", type=str, default=".jpg")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--conf_thres", type=float, default=0.2)
    args = parser.parse_args()

    if len(args.subs) == 0:
        args.subs = sorted([d for d in os.listdir(join(args.path, "images")) if os.path.isdir(join(args.path, "images", d))])

    for sub in args.subs:
        img_dir = join(args.path, "images", sub)
        ann_dir = join(args.path, args.annot, sub)
        out_dir = join(args.path, args.out, sub)
        os.makedirs(out_dir, exist_ok=True)
        ann_files = sorted(glob(join(ann_dir, "*.json")))
        for annf in ann_files:
            stem = basename(annf).replace(".json", "")
            if stem.isdigit():
                idx = int(stem)
                if args.start is not None and idx < args.start:
                    continue
                if args.end is not None and idx >= args.end:
                    continue
            imgf = join(img_dir, stem + args.ext)
            img = cv2.imread(imgf)
            if img is None:
                continue
            data = read_json(annf)
            annots = data.get("annots", []) if isinstance(data, dict) else []
            for ann in annots:
                draw_one(img, ann, conf_thres=args.conf_thres)
            cv2.imwrite(join(out_dir, stem + ".jpg"), img)
        print(f"{sub}: done")
    print("Output:", join(args.path, args.out))


if __name__ == "__main__":
    main()

