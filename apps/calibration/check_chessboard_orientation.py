import argparse
import json
import os
from glob import glob
from os.path import join

import cv2
import numpy as np


MODES = ["identity", "flip_lr", "flip_ud", "flip_both"]


def parse_pattern(s):
    if s is None:
        return None
    a, b = s.split(",")
    return int(a), int(b)


def parse_csv(s):
    if s is None or s.strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def parse_apply_map(s):
    out = {}
    if s is None or s.strip() == "":
        return out
    for item in s.split(","):
        item = item.strip()
        if item == "":
            continue
        if ":" not in item:
            raise ValueError(f"Invalid item in --apply_map: {item}")
        cam, mode = item.split(":", 1)
        cam = cam.strip()
        mode = mode.strip()
        if mode not in MODES:
            raise ValueError(f"Invalid mode '{mode}' for camera '{cam}'")
        out[cam] = mode
    return out


def load_apply_json(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for cam, mode in data.items():
        if mode not in MODES:
            raise ValueError(f"Invalid mode '{mode}' for camera '{cam}' in {path}")
        out[cam] = mode
    return out


def resolve_frame_path(root, cam, frame, ext):
    cam_dir = join(root, cam)
    if not os.path.isdir(cam_dir):
        return None
    if frame is None:
        imgs = sorted(glob(join(cam_dir, f"*{ext}")))
        return imgs[0] if len(imgs) > 0 else None
    candidates = []
    frame = str(frame)
    candidates.append(frame)
    if frame.isdigit():
        n = int(frame)
        candidates.extend(
            [f"{n:06d}", f"{n:05d}", f"{n:04d}", f"{n:03d}", f"{n:02d}", f"{n:01d}"]
        )
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        p = join(cam_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def remap_k2d(k2d, pattern, mode):
    w, h = pattern
    n = w * h
    if k2d.shape[0] != n:
        raise ValueError(f"keypoints size {k2d.shape[0]} does not match pattern {pattern}")
    arr = k2d.reshape(h, w, -1)
    if mode == "identity":
        arr2 = arr
    elif mode == "flip_lr":
        arr2 = arr[:, ::-1, :]
    elif mode == "flip_ud":
        arr2 = arr[::-1, :, :]
    elif mode == "flip_both":
        arr2 = arr[::-1, ::-1, :]
    else:
        raise ValueError(mode)
    return arr2.reshape(n, -1)


def draw_mode_panel(img, k2d, pattern, title):
    show = img.copy()
    w, h = pattern
    n = w * h

    pts = k2d[:, :2]
    conf = k2d[:, 2]
    valid = conf > 0.01
    for i in range(n):
        if not valid[i]:
            continue
        x, y = int(round(float(pts[i, 0]))), int(round(float(pts[i, 1])))
        cv2.circle(show, (x, y), 2, (255, 255, 0), -1)

    def draw_idx(i, color, text):
        if i < 0 or i >= n:
            return
        if not valid[i]:
            return
        x, y = int(round(float(pts[i, 0]))), int(round(float(pts[i, 1])))
        cv2.circle(show, (x, y), 6, color, 2)
        cv2.putText(
            show,
            text,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    i0 = 0
    i1 = 1 if w * h > 1 else -1
    iw = w if w < n else -1
    ilast = n - 1

    draw_idx(i0, (0, 0, 255), "0")
    draw_idx(i1, (0, 255, 0), "1")
    draw_idx(iw, (255, 0, 0), "w")
    draw_idx(ilast, (0, 255, 255), "last")

    if i0 >= 0 and i1 >= 0 and valid[i0] and valid[i1]:
        p0 = tuple(np.round(pts[i0]).astype(int).tolist())
        p1 = tuple(np.round(pts[i1]).astype(int).tolist())
        cv2.arrowedLine(show, p0, p1, (0, 255, 0), 2, tipLength=0.15)
    if i0 >= 0 and iw >= 0 and valid[i0] and valid[iw]:
        p0 = tuple(np.round(pts[i0]).astype(int).tolist())
        pw = tuple(np.round(pts[iw]).astype(int).tolist())
        cv2.arrowedLine(show, p0, pw, (255, 0, 0), 2, tipLength=0.15)

    cv2.putText(
        show,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return show


def make_montage(panels):
    h, w = panels[0].shape[:2]
    rows = []
    for r in range(0, len(panels), 2):
        a = panels[r]
        b = panels[r + 1] if r + 1 < len(panels) else np.zeros_like(a)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h))
        if b.shape[:2] != (h, w):
            b = cv2.resize(b, (w, h))
        rows.append(cv2.hconcat([a, b]))
    return cv2.vconcat(rows)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root")
    parser.add_argument("--image", type=str, default="images")
    parser.add_argument("--chessboard", type=str, default="chessboard")
    parser.add_argument("--ext", type=str, default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument("--frame", type=str, default=None, help="frame id, e.g. 008500")
    parser.add_argument("--pattern", type=str, default=None, help="w,h ; fallback to json pattern")
    parser.add_argument("--cams", type=str, default=None, help="camera list csv")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output folder for visualization (default: <path>/orientation_check)",
    )
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument(
        "--apply_map",
        type=str,
        default=None,
        help="cam:mode pairs, e.g. D04:flip_both,D06:flip_both",
    )
    parser.add_argument(
        "--apply_json",
        type=str,
        default=None,
        help="json file {camera_name: mode}",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="backup original json as .bak before applying remap",
    )
    args = parser.parse_args()

    if args.out is None:
        args.out = join(args.path, "orientation_check")

    image_root = join(args.path, args.image)
    chess_root = join(args.path, args.chessboard)
    cams = sorted([d for d in os.listdir(image_root) if os.path.isdir(join(image_root, d))])
    cam_filter = set(parse_csv(args.cams))
    if len(cam_filter) > 0:
        cams = [c for c in cams if c in cam_filter]
    if len(cams) == 0:
        raise RuntimeError(f"No cameras found in {image_root}")

    pattern_cli = parse_pattern(args.pattern)
    apply_map = {}
    apply_map.update(load_apply_json(args.apply_json))
    apply_map.update(parse_apply_map(args.apply_map))

    report = {}
    for cam in cams:
        img_path = resolve_frame_path(image_root, cam, args.frame, args.ext)
        if img_path is None:
            print(f"[skip] {cam}: cannot find image frame")
            continue
        frame_name = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = join(chess_root, cam, frame_name + ".json")
        if not os.path.exists(ann_path):
            print(f"[skip] {cam}: missing {ann_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[skip] {cam}: cannot read image {img_path}")
            continue

        with open(ann_path, "r") as f:
            data = json.load(f)
        k2d = np.asarray(data["keypoints2d"], dtype=np.float32)

        if pattern_cli is not None:
            pattern = pattern_cli
        else:
            if "pattern" not in data:
                print(f"[skip] {cam}: missing pattern in json and no --pattern provided")
                continue
            pattern = tuple(int(v) for v in data["pattern"])
        n_expected = int(pattern[0] * pattern[1])
        if k2d.shape[0] != n_expected:
            print(
                f"[skip] {cam}: keypoints={k2d.shape[0]} mismatch pattern={pattern}({n_expected})"
            )
            continue

        vis_paths = {}
        if not args.no_vis:
            panels = []
            for mode in MODES:
                k2d_mode = remap_k2d(k2d, pattern, mode)
                p = draw_mode_panel(img, k2d_mode, pattern, mode)
                panels.append(p)
                single_out = join(args.out, cam, f"{frame_name}_{mode}.jpg")
                os.makedirs(os.path.dirname(single_out), exist_ok=True)
                cv2.imwrite(single_out, p)
                vis_paths[mode] = single_out
            montage = make_montage(panels)
            montage_path = join(args.out, cam, f"{frame_name}_montage.jpg")
            cv2.imwrite(montage_path, montage)
            vis_paths["montage"] = montage_path

        selected_mode = apply_map.get(cam, "identity")
        if selected_mode not in MODES:
            raise ValueError(f"Invalid mode '{selected_mode}' for {cam}")

        k2d_sel = remap_k2d(k2d, pattern, selected_mode)
        valid = k2d_sel[:, 2] > 0.01
        p0 = (
            [float(k2d_sel[0, 0]), float(k2d_sel[0, 1])]
            if valid[0]
            else [None, None]
        )

        if cam in apply_map:
            if args.backup:
                bak = ann_path + ".bak"
                if not os.path.exists(bak):
                    with open(bak, "w") as f:
                        json.dump(data, f, indent=2)
            data["keypoints2d"] = k2d_sel.tolist()
            data["orientation_mode"] = selected_mode
            save_json(ann_path, data)
            print(f"[apply] {cam}: {selected_mode} -> {ann_path}")
        else:
            print(f"[check] {cam}: current mode=identity, frame={frame_name}")

        report[cam] = {
            "frame": frame_name,
            "pattern": [int(pattern[0]), int(pattern[1])],
            "selected_mode": selected_mode,
            "point0_xy": p0,
            "valid_points": int(valid.sum()),
            "ann_path": ann_path,
            "vis": vis_paths,
        }

    os.makedirs(args.out, exist_ok=True)
    report_path = join(args.out, "orientation_report.json")
    save_json(report_path, report)
    print(f"[done] report: {report_path}")
    print("[tip] check montage first, then rerun with --apply_map or --apply_json to write remap.")


if __name__ == "__main__":
    main()
