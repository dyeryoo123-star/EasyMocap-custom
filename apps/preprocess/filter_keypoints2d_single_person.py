import argparse
import os
from glob import glob
from os.path import basename, join

import numpy as np

from easymocap.mytools.file_utils import read_json, save_json


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = aa + bb - inter + 1e-6
    return inter / union


def bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def select_one(annots, prev_bbox=None):
    if len(annots) == 0:
        return None
    scores = []
    for a in annots:
        b = a.get("bbox", [0, 0, 0, 0, 0])
        conf = float(b[4]) if len(b) > 4 else 0.0
        area = bbox_area(b)
        if prev_bbox is None:
            score = area * (0.5 + conf)
        else:
            score = 0.7 * iou_xyxy(prev_bbox, b) + 0.2 * conf + 0.1 * np.log(area + 1.0)
        scores.append(score)
    idx = int(np.argmax(scores))
    return annots[idx]


def clean_keypoints(kpts, conf_thres=0.2):
    arr = np.asarray(kpts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return kpts
    arr[arr[:, 2] < conf_thres, :] = 0.0
    return arr.tolist()


def process_sub(in_dir, out_dir, conf_thres=0.2):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob(join(in_dir, "*.json")))
    prev_bbox = None
    for fn in files:
        data = read_json(fn)
        annots = data.get("annots", []) if isinstance(data, dict) else []
        best = select_one(annots, prev_bbox=prev_bbox)
        if best is not None:
            if "keypoints" in best:
                best["keypoints"] = clean_keypoints(best["keypoints"], conf_thres=conf_thres)
            best["personID"] = 0
            data["annots"] = [best]
            prev_bbox = best.get("bbox", prev_bbox)
        else:
            data["annots"] = []
        save_json(join(out_dir, basename(fn)), data)
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Dataset root")
    parser.add_argument("--annot_in", type=str, default="annots_upright_v1")
    parser.add_argument("--annot_out", type=str, default="annots_upright_v1_clean")
    parser.add_argument("--subs", nargs="+", default=[])
    parser.add_argument("--conf_thres", type=float, default=0.2)
    args = parser.parse_args()

    in_root = join(args.path, args.annot_in)
    out_root = join(args.path, args.annot_out)
    if len(args.subs) == 0:
        args.subs = sorted([d for d in os.listdir(in_root) if os.path.isdir(join(in_root, d))])
    for sub in args.subs:
        n = process_sub(join(in_root, sub), join(out_root, sub), conf_thres=args.conf_thres)
        print(f"{sub}: {n} frames")
    print("Output:", out_root)


if __name__ == "__main__":
    main()

