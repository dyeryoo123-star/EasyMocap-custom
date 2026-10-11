"""
Rotate frames by 180 degrees for pose detection, then map detections back
to the original image coordinates.
"""

import argparse
import os
import shutil
import subprocess
import shutil as _shutil
from glob import glob
from os.path import basename, dirname, exists, isdir, join

import cv2

from easymocap.mytools.file_utils import read_json, save_json


def list_subs(path, subs):
    if len(subs) == 0:
        subs = sorted(os.listdir(join(path, "images")))
    subs = [s for s in subs if isdir(join(path, "images", s))]
    return subs


def list_images(image_dir, ext):
    files = sorted(glob(join(image_dir, "*" + ext)))
    return files


def select_by_range(files, start, end, step):
    if start is None and end is None:
        return files
    names = [basename(f) for f in files]
    out = []
    for fn in names:
        stem = os.path.splitext(fn)[0]
        if not stem.isdigit():
            continue
        idx = int(stem)
        if start is not None and idx < start:
            continue
        if end is not None and idx >= end:
            continue
        if step > 1 and start is not None and (idx - start) % step != 0:
            continue
        out.append(join(dirname(files[0]), fn))
    return out


def rotate_images(src_dir, dst_dir, ext, start=None, end=None, step=1):
    os.makedirs(dst_dir, exist_ok=True)
    files = list_images(src_dir, ext)
    files = select_by_range(files, start, end, step)
    for src in files:
        dst = join(dst_dir, basename(src))
        img = cv2.imread(src)
        if img is None:
            continue
        rot = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(dst, rot)
    return len(files)


def _flip_point(x, y, w, h):
    return (w - 1 - x), (h - 1 - y)


def _flip_bbox(bbox, w, h):
    if len(bbox) < 4:
        return bbox
    x1, y1, x2, y2 = bbox[:4]
    nx1, ny1 = _flip_point(x2, y2, w, h)
    nx2, ny2 = _flip_point(x1, y1, w, h)
    out = [float(nx1), float(ny1), float(nx2), float(ny2)]
    if len(bbox) > 4:
        out.extend(bbox[4:])
    return out


def _flip_kpts(kpts, w, h):
    out = []
    for item in kpts:
        if len(item) < 2:
            out.append(item)
            continue
        x, y = item[0], item[1]
        nx, ny = _flip_point(x, y, w, h)
        if len(item) > 2:
            out.append([float(nx), float(ny), item[2]])
        else:
            out.append([float(nx), float(ny)])
    return out


def remap_annots_file(src_json, dst_json):
    data = read_json(src_json)
    if isinstance(data, dict):
        ann_list = data.get("annots", [])
        w = data.get("width", None)
        h = data.get("height", None)
        if w is None or h is None:
            save_json(dst_json, data)
            return
        for ann in ann_list:
            for key in ["bbox", "bbox_handl2d", "bbox_handr2d", "bbox_face2d"]:
                if key in ann:
                    ann[key] = _flip_bbox(ann[key], w, h)
            for key in ["keypoints", "keypoints2d", "handl2d", "handr2d", "face2d"]:
                if key in ann:
                    ann[key] = _flip_kpts(ann[key], w, h)
        save_json(dst_json, data)
    else:
        # Fallback for list format
        save_json(dst_json, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--subs", nargs="+", default=[])
    parser.add_argument("--ext", type=str, default=".jpg")
    parser.add_argument("--mode", type=str, default="openposecrop")
    parser.add_argument("--annot_out", type=str, default="annots_upright")
    parser.add_argument("--annot_tmp", type=str, default="annots_tmp_upright")
    parser.add_argument("--tmp_dataset", type=str, default="_tmp_upright_dataset")
    parser.add_argument("--openpose", type=str, default=os.environ.get("openpose", "openpose"))
    parser.add_argument("--openpose_res", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--hand", action="store_true")
    parser.add_argument("--face", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    subs = list_subs(args.path, args.subs)
    if len(subs) == 0:
        raise RuntimeError("No valid camera subfolders found under images/")

    # Basic dependency checks for selected detector mode.
    if args.mode in ["yolo", "yolo-hrnet", "openposecrop"]:
        yolo_w = join("data", "models", "yolov4.weights")
        if not exists(yolo_w):
            raise FileNotFoundError(
                "Missing YOLO weights: {}\n"
                "Please prepare this file first, or switch to --mode openpose (needs OpenPose).".format(yolo_w)
            )
    if args.mode in ["openpose", "openposecrop", "feet", "feetcrop"]:
        # openpose argument can be an executable path, install root, or command in PATH.
        openpose_ok = exists(args.openpose) or (_shutil.which(args.openpose) is not None)
        if not openpose_ok:
            raise FileNotFoundError(
                "OpenPose not found: '{}'\n"
                "Please pass a valid --openpose path/command, e.g. --openpose /path/to/openpose".format(args.openpose)
            )

    tmp_root = join(args.path, args.tmp_dataset)
    tmp_images = join(tmp_root, "images")
    tmp_annots = join(tmp_root, args.annot_tmp)
    if exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_images, exist_ok=True)

    print("[1/4] Rotating images for detection...")
    for sub in subs:
        src_dir = join(args.path, "images", sub)
        dst_dir = join(tmp_images, sub)
        n = rotate_images(src_dir, dst_dir, args.ext, start=args.start, end=args.end, step=args.step)
        print(f"  {sub}: {n} frames")

    try:
        print("[2/4] Running 2D detection on upright images...")
        cmd = [
            "python3", "apps/preprocess/extract_keypoints.py", tmp_root,
            "--mode", args.mode,
            "--annot", args.annot_tmp,
            "--openpose", args.openpose,
            "--openpose_res", str(args.openpose_res),
            "--ext", args.ext,
            "--subs",
        ] + subs
        if args.hand:
            cmd.append("--hand")
        if args.face:
            cmd.append("--face")
        if args.force:
            cmd.append("--force")
        print("[cmd]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        print("[3/4] Mapping detections back to original image coordinates...")
        out_root = join(args.path, args.annot_out)
        for sub in subs:
            in_dir = join(tmp_annots, sub)
            out_dir = join(out_root, sub)
            os.makedirs(out_dir, exist_ok=True)
            for src_json in sorted(glob(join(in_dir, "*.json"))):
                dst_json = join(out_dir, basename(src_json))
                remap_annots_file(src_json, dst_json)
            print(f"  {sub}: {len(glob(join(in_dir, '*.json')))} annots")

        print("[4/4] Done.")
        print("Output annots:", out_root)
    except subprocess.CalledProcessError as e:
        print("[Error] 2D detector failed with non-zero exit status.")
        print("        Please check detector dependencies:")
        print("        - openposecrop/yolo-hrnet need data/models/yolov4.weights")
        print("        - openpose/openposecrop need a valid --openpose path/command")
        print("        - mediapipe may fail in headless OpenGL environments")
        raise e
    finally:
        if not args.keep_tmp and exists(tmp_root):
            shutil.rmtree(tmp_root)
            print("Temp folder removed:", tmp_root)
        elif args.keep_tmp:
            print("Temp folder kept:", tmp_root)


if __name__ == "__main__":
    main()
