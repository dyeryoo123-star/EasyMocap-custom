import argparse
import json
import os
from glob import glob
from os.path import join

import cv2
import numpy as np


ARUCO_DICT = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
}

ARUCO_CAPACITY = {
    "4X4_50": 50,
    "4X4_100": 100,
    "4X4_250": 250,
    "4X4_1000": 1000,
    "5X5_100": 100,
    "5X5_250": 250,
    "5X5_1000": 1000,
    "6X6_250": 250,
    "6X6_1000": 1000,
    "7X7_1000": 1000,
}

BOARD_TEMPLATES = {
    # User-provided custom values from CLI
    "custom": None,
    # EasyMocap historical defaults
    "easymocap_default": {
        "long": 6,
        "short": 4,
        "square_length": 0.128,
        "marker_length": 0.10,
        "aruco_type": "4X4_50",
    },
    # A more likely setup for the current capture board
    "lab_charuco_12x8": {
        "long": 12,
        "short": 8,
        "square_length": 0.16,
        "marker_length": 0.10,
        "aruco_type": "4X4_1000",
    },
}


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_csv_list(s):
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def parse_float_list(s):
    out = []
    for x in parse_csv_list(s):
        out.append(float(x))
    return out


def load_roi_map(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--roi_json must be a json object: {cam: roi}")
    return data


def load_id_layout(path):
    if path is None:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--id_layout_json must be a json object")
    return data


def parse_frame_filter(s):
    items = parse_csv_list(s)
    if len(items) == 0:
        return None
    raw = set(items)
    as_int = set()
    for x in items:
        try:
            as_int.add(int(x))
        except ValueError:
            pass
    return {"raw": raw, "as_int": as_int}


def frame_match(frame_name, frame_filter):
    if frame_filter is None:
        return True
    if frame_name in frame_filter["raw"]:
        return True
    try:
        frame_i = int(frame_name)
    except ValueError:
        return False
    return frame_i in frame_filter["as_int"]


def match_roi_key(cam, roi_map):
    if cam in roi_map:
        return cam
    prefix = cam.split("_")[0]
    if prefix in roi_map:
        return prefix
    for key in roi_map.keys():
        if cam.startswith(key):
            return key
    return None


def parse_roi_entry(entry, fmt, width, height):
    if isinstance(entry, dict):
        if all(k in entry for k in ["x", "y", "w", "h"]):
            x1 = int(round(float(entry["x"])))
            y1 = int(round(float(entry["y"])))
            x2 = x1 + int(round(float(entry["w"])))
            y2 = y1 + int(round(float(entry["h"])))
        elif all(k in entry for k in ["x1", "y1", "x2", "y2"]):
            x1 = int(round(float(entry["x1"])))
            y1 = int(round(float(entry["y1"])))
            x2 = int(round(float(entry["x2"])))
            y2 = int(round(float(entry["y2"])))
        else:
            raise ValueError("roi dict must be {x,y,w,h} or {x1,y1,x2,y2}")
    elif isinstance(entry, (list, tuple)) and len(entry) == 4:
        a, b, c, d = [int(round(float(v))) for v in entry]
        if fmt == "xywh":
            x1, y1, x2, y2 = a, b, a + c, b + d
        else:
            x1, y1, x2, y2 = a, b, c, d
    else:
        raise ValueError("roi entry must be list[4] or dict")

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def get_cam_roi(cam, roi_map, roi_format, width, height):
    if len(roi_map) == 0:
        return None
    key = match_roi_key(cam, roi_map)
    if key is None:
        return None
    return parse_roi_entry(roi_map[key], roi_format, width, height)


def prepare_detect_image(img, cam, args):
    height, width = img.shape[:2]
    roi = get_cam_roi(cam, args.roi_map, args.roi_format, width, height)
    x1, y1 = 0, 0
    work = img
    if roi is not None:
        x1, y1, x2, y2 = roi
        work = img[y1:y2, x1:x2]
    scale = args.roi_upscale if roi is not None else 1.0
    if scale != 1.0:
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return work, {"x1": x1, "y1": y1, "scale": scale, "roi": roi}


def map_corners_to_full_image(corners2d, tf):
    if corners2d is None:
        return None
    out = corners2d.copy()
    if tf["scale"] != 1.0:
        out = out / float(tf["scale"])
    out[:, 0] += float(tf["x1"])
    out[:, 1] += float(tf["y1"])
    return out


def map_marker_corners_to_full_image(marker_corners, tf):
    if marker_corners is None:
        return None
    out = []
    for c in marker_corners:
        m = c.copy()
        if tf["scale"] != 1.0:
            m = m / float(tf["scale"])
        m[:, :, 0] += float(tf["x1"])
        m[:, :, 1] += float(tf["y1"])
        out.append(m)
    return out


def create_board(aruco_type, long, short, square_length, marker_length, custom_ids=None):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    ids = None
    if custom_ids is not None:
        ids = np.asarray(custom_ids, dtype=np.int32).reshape(-1, 1)
    if hasattr(cv2.aruco, "CharucoBoard"):
        if ids is None:
            board = cv2.aruco.CharucoBoard(
                (long, short), square_length, marker_length, dictionary
            )
        else:
            board = cv2.aruco.CharucoBoard(
                (long, short), square_length, marker_length, dictionary, ids
            )
    else:
        if ids is not None:
            raise RuntimeError("Custom ids requires OpenCV with CharucoBoard constructor.")
        board = cv2.aruco.CharucoBoard_create(
            squaresX=long,
            squaresY=short,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=dictionary,
        )
    if hasattr(board, "getChessboardCorners"):
        corners3d = board.getChessboardCorners()
    else:
        corners3d = board.chessboardCorners
    return dictionary, board, np.asarray(corners3d, dtype=np.float32)


def build_marker_lookup(board):
    ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    obj = board.getObjPoints()
    lookup = {}
    for i, marker_id in enumerate(ids):
        lookup[int(marker_id)] = np.asarray(obj[i], dtype=np.float32)
    return lookup


def create_board_candidates(
    aruco_type,
    long,
    short,
    square_length,
    marker_length,
    marker_ratios=None,
    custom_ids=None,
):
    if aruco_type == "auto":
        names = sorted(ARUCO_DICT.keys())
    else:
        names = [aruco_type]
    if marker_length > 0:
        ratios = [float(marker_length) / float(square_length)]
    else:
        ratios = marker_ratios if marker_ratios is not None else [0.6]
    ratios = sorted(set([float(r) for r in ratios]))
    candidates = []
    corners3d_ref = None
    for name in names:
        for ratio in ratios:
            cur_marker_length = float(square_length) * ratio
            if cur_marker_length <= 0 or cur_marker_length >= float(square_length):
                continue
            dictionary, board, corners3d = create_board(
                name,
                long,
                short,
                square_length,
                cur_marker_length,
                custom_ids=custom_ids,
            )
            if corners3d_ref is None:
                corners3d_ref = corners3d
            candidates.append(
                {
                    "name": f"{name}@r{ratio:.3f}",
                    "aruco_type": name,
                    "ratio": ratio,
                    "dictionary": dictionary,
                    "board": board,
                    "corners3d": corners3d,
                    "marker_lookup": build_marker_lookup(board),
                    "marker_length": cur_marker_length,
                    "square_length": square_length,
                }
            )
    if len(candidates) == 0:
        raise RuntimeError("No valid board candidate generated.")
    return candidates, corners3d_ref


def create_detector_params(detect_inverted=False):
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = min(params.minMarkerPerimeterRate, 0.01)
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = bool(detect_inverted)
    return params


def detect_markers(gray, dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params
        )
    return corners, ids, rejected


def detect_markers_multiscale(gray, dictionary, params, scales):
    best = (None, None, -1, 1.0)
    for scale in scales:
        if scale == 1.0:
            work = gray
        else:
            work = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        corners, ids, _ = detect_markers(work, dictionary, params)
        n = 0 if ids is None else int(len(ids))
        if n > best[2]:
            if n > 0 and scale != 1.0:
                corners = [c / float(scale) for c in corners]
            best = (corners, ids, n, scale)
    return best


def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.asarray([tl, tr, br, bl], dtype=np.float32)


def estimate_board_quad(gray):
    h, w = gray.shape[:2]
    area_min = 0.01 * float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
    )
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = -1.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < area_min:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)
    if best is None:
        if len(cnts) == 0:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < area_min:
            return None
        rect = cv2.minAreaRect(c)
        best = cv2.boxPoints(rect).astype(np.float32)
    return order_quad(best)


def marker_slot_cells(long, short, aruco_type):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    board = cv2.aruco.CharucoBoard((long, short), 1.0, 0.6, dictionary)
    obj = board.getObjPoints()
    cells = []
    for m in obj:
        center = np.asarray(m, dtype=np.float32)[:, :2].mean(axis=0)
        cx = int(np.floor(float(center[0]) + 1e-6))
        cy = int(np.floor(float(center[1]) + 1e-6))
        cells.append((cx, cy))
    default_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1).tolist()
    return cells, default_ids


def assign_markers_to_cells(marker_corners, marker_ids, quad, long, short):
    dst = np.asarray(
        [[0, 0], [long, 0], [long, short], [0, short]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst)
    centers = np.asarray(
        [c.reshape(-1, 2).mean(axis=0) for c in marker_corners], dtype=np.float32
    ).reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(centers, H).reshape(-1, 2)
    ids = marker_ids.reshape(-1).astype(np.int32)

    parity_records = {0: [], 1: []}
    for idx, (u, v) in enumerate(uv):
        cx = int(np.floor(float(u)))
        cy = int(np.floor(float(v)))
        if cx < 0 or cx >= long or cy < 0 or cy >= short:
            continue
        parity = (cx + cy) % 2
        parity_records[parity].append((int(ids[idx]), cx, cy))

    p0 = len(parity_records[0])
    p1 = len(parity_records[1])
    use = 0 if p0 >= p1 else 1
    return parity_records[use]


def reconstruct_id_layout(args):
    image_root = join(args.path, args.image)
    cams = list_cameras(image_root)
    if len(cams) == 0:
        raise RuntimeError(f"No camera folder found in {image_root}")

    cam = args.recon_cam if args.recon_cam is not None else cams[0]
    if cam not in cams:
        raise RuntimeError(f"recon cam {cam} not found in {image_root}")

    frame_filter = parse_frame_filter(args.frames)
    frame_map = collect_images(
        image_root, cam, args.ext, frame_filter=frame_filter, step=max(1, args.step)
    )
    frames = sorted(frame_map.keys())
    if len(frames) == 0:
        raise RuntimeError(f"No frames found for recon cam {cam}")

    scales = [float(x) for x in args.scales.split(",") if x.strip() != ""]
    params = create_detector_params(detect_inverted=args.detect_inverted)
    dict_names = sorted(ARUCO_DICT.keys()) if args.aruco_type == "auto" else [args.aruco_type]

    dict_score = {name: 0 for name in dict_names}
    for frame in frames:
        img = cv2.imread(frame_map[frame])
        if img is None:
            continue
        work, _ = prepare_detect_image(img, cam, args)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        for name in dict_names:
            dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[name])
            _, ids, n, _ = detect_markers_multiscale(gray, dictionary, params, scales)
            dict_score[name] += n
    best_dict = sorted(dict_names, key=lambda n: dict_score[n], reverse=True)[0]
    print("[recon] dictionary ranking:")
    for name in sorted(dict_names, key=lambda n: dict_score[n], reverse=True):
        print(f"  {name}: markers={dict_score[name]}")
    print(f"[recon] selected dictionary: {best_dict}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[best_dict])
    votes = {}
    used_frames = 0
    debug_vis = None

    for frame in frames:
        img = cv2.imread(frame_map[frame])
        if img is None:
            continue
        work, tf = prepare_detect_image(img, cam, args)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, n, _ = detect_markers_multiscale(
            gray, dictionary, params, scales
        )
        if marker_ids is None or n < args.recon_min_markers:
            continue
        quad = estimate_board_quad(gray)
        if quad is None:
            continue
        triples = assign_markers_to_cells(
            marker_corners, marker_ids, quad, args.long, args.short
        )
        if len(triples) < args.recon_min_markers:
            continue
        used_frames += 1
        for marker_id, cx, cy in triples:
            key = (int(marker_id), int(cx), int(cy))
            votes[key] = votes.get(key, 0) + 1

        if debug_vis is None:
            vis = work.copy()
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 255), 2)
            for marker_id, cx, cy in triples:
                center = (int((cx + 0.5) / args.long * vis.shape[1]), int((cy + 0.5) / args.short * vis.shape[0]))
                cv2.putText(
                    vis,
                    f"{marker_id}->{cx},{cy}",
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            debug_vis = vis

    if used_frames == 0 or len(votes) == 0:
        raise RuntimeError("reconstruct failed: no usable frame with markers and board quad")

    triples = sorted(
        [(cnt, m, x, y) for (m, x, y), cnt in votes.items()],
        key=lambda t: t[0],
        reverse=True,
    )
    id_to_cell = {}
    used_cells = set()
    for cnt, marker_id, cx, cy in triples:
        if marker_id in id_to_cell:
            continue
        if (cx, cy) in used_cells:
            continue
        id_to_cell[marker_id] = (cx, cy, cnt)
        used_cells.add((cx, cy))

    grid = [[-1 for _ in range(args.long)] for _ in range(args.short)]
    for marker_id, (cx, cy, _) in id_to_cell.items():
        if 0 <= cx < args.long and 0 <= cy < args.short:
            grid[cy][cx] = int(marker_id)

    slot_cells, default_ids = marker_slot_cells(args.long, args.short, best_dict)
    cell_to_id = {}
    for marker_id, (cx, cy, _) in id_to_cell.items():
        cell_to_id[(cx, cy)] = int(marker_id)

    cap = ARUCO_CAPACITY[best_dict]
    used_marker_ids = set(cell_to_id.values())
    fill_pool = [i for i in range(cap) if i not in used_marker_ids]
    ids_board = []
    missing_slots = 0
    for idx, cell in enumerate(slot_cells):
        if cell in cell_to_id:
            ids_board.append(int(cell_to_id[cell]))
        else:
            missing_slots += 1
            if len(fill_pool) > 0:
                ids_board.append(int(fill_pool.pop(0)))
            else:
                ids_board.append(int(default_ids[idx]))

    out_json = args.recon_out if args.recon_out is not None else join(args.out, "id_layout.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    record = {
        "long": int(args.long),
        "short": int(args.short),
        "aruco_type": best_dict,
        "board_order_ids": ids_board,
        "grid_ids": grid,
        "frames_used": int(used_frames),
        "assigned_cells": int(len(used_cells)),
        "total_slots": int(len(slot_cells)),
        "missing_slots_filled": int(missing_slots),
        "recon_cam": cam,
    }
    save_json(out_json, record)
    print(f"[recon] frames used: {used_frames}")
    print(f"[recon] assigned cells: {len(used_cells)}/{len(slot_cells)}")
    print(f"[recon] saved layout: {out_json}")

    if debug_vis is not None:
        debug_path = join(os.path.dirname(out_json), "recon_debug.jpg")
        cv2.imwrite(debug_path, debug_vis)
        print(f"[recon] saved debug: {debug_path}")

    return out_json


def detect_charuco(gray, dictionary, board, params, scales, min_corners):
    for scale in scales:
        if scale == 1.0:
            work = gray
        else:
            work = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        marker_corners, marker_ids, _ = detect_markers(work, dictionary, params)
        if marker_ids is None or len(marker_ids) == 0:
            continue

        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=work,
            board=board,
        )
        if not ret or ch_ids is None:
            continue

        ch_ids = ch_ids.reshape(-1).astype(np.int32)
        ch_corners = ch_corners.reshape(-1, 2).astype(np.float32)
        if scale != 1.0:
            ch_corners = ch_corners / scale
        if ch_ids.shape[0] < min_corners:
            continue
        return ch_ids, ch_corners
    return None, None


def match_aruco_marker_points(marker_corners, marker_ids, marker_lookup):
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, None
    ids = marker_ids.reshape(-1).astype(np.int32)
    obj_list = []
    img_list = []
    keep_ids = []
    seen = set()
    for i, marker_id in enumerate(ids):
        if marker_id in seen:
            continue
        if int(marker_id) not in marker_lookup:
            continue
        seen.add(int(marker_id))
        obj = marker_lookup[int(marker_id)]
        img = np.asarray(marker_corners[i], dtype=np.float32).reshape(4, 2)
        obj_list.append(obj)
        img_list.append(img)
        keep_ids.append(int(marker_id))
    if len(obj_list) == 0:
        return None, None, None
    obj = np.concatenate(obj_list, axis=0).astype(np.float32)
    img = np.concatenate(img_list, axis=0).astype(np.float32)
    keep_ids = np.asarray(keep_ids, dtype=np.int32)
    return obj, img, keep_ids


def detect_aruco_points(gray, dictionary, marker_lookup, params, scales):
    marker_corners, marker_ids, n_markers, _ = detect_markers_multiscale(
        gray, dictionary, params, scales
    )
    if marker_ids is None or n_markers <= 0:
        return None, None, None, None
    obj, img, keep_ids = match_aruco_marker_points(
        marker_corners, marker_ids, marker_lookup
    )
    if obj is None:
        return marker_corners, marker_ids, None, None
    return marker_corners, marker_ids, obj, img


def to_chessboard_record(corners3d, ids, corners2d, long, short, square_length):
    k2d = np.zeros((corners3d.shape[0], 3), dtype=np.float32)
    if ids is not None and corners2d is not None and ids.shape[0] > 0:
        k2d[ids, :2] = corners2d
        k2d[ids, 2] = 1.0
    return {
        "keypoints3d": corners3d.tolist(),
        "keypoints2d": k2d.tolist(),
        "ids": [] if ids is None else ids.tolist(),
        "pattern": [long - 1, short - 1],
        "grid_size": square_length,
        "visited": True,
    }


def write_matrix(f, key, value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    f.write(f"{key}: !!opencv-matrix\n")
    f.write(f"  rows: {value.shape[0]}\n")
    f.write(f"  cols: {value.shape[1]}\n")
    f.write("  dt: d\n")
    data = ", ".join([f"{x:.6f}" for x in value.reshape(-1)])
    f.write(f"  data: [{data}]\n")


def write_intri(path, intri):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    camnames = sorted(intri.keys())
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for cam in camnames:
            f.write(f'  - "{cam}"\n')
        for cam in camnames:
            write_matrix(f, f"K_{cam}", intri[cam]["K"])
            write_matrix(f, f"dist_{cam}", intri[cam]["dist"])


def write_extri(path, extri):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    camnames = sorted(extri.keys())
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for cam in camnames:
            f.write(f'  - "{cam}"\n')
        for cam in camnames:
            write_matrix(f, f"R_{cam}", extri[cam]["rvec"])
            write_matrix(f, f"Rot_{cam}", extri[cam]["R"])
            write_matrix(f, f"T_{cam}", extri[cam]["T"])


def read_names(fs_node):
    out = []
    for i in range(fs_node.size()):
        s = fs_node.at(i).string()
        if s == "":
            s = str(int(fs_node.at(i).real()))
        out.append(s)
    return out


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names = read_names(fs.getNode("names"))
    intri = {}
    for cam in names:
        K = fs.getNode(f"K_{cam}").mat()
        dist = fs.getNode(f"dist_{cam}").mat()
        intri[cam] = {"K": K, "dist": dist}
    fs.release()
    return intri


def list_cameras(image_root):
    cams = sorted(os.listdir(image_root))
    cams = [c for c in cams if os.path.isdir(join(image_root, c))]
    return cams


def collect_images(image_root, cam, ext, frame_filter=None, step=1):
    imgnames = sorted(glob(join(image_root, cam, f"*{ext}")))
    frame_map = {}
    kept = 0
    for img in imgnames:
        frame = os.path.splitext(os.path.basename(img))[0]
        if not frame_match(frame, frame_filter):
            continue
        if step > 1 and (kept % step) != 0:
            kept += 1
            continue
        frame_map[frame] = img
        kept += 1
    return frame_map


def sample_probe_frames(frame_map, max_frames):
    frames = sorted(frame_map.keys())
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, max_frames, dtype=np.int32)
    return [frames[int(i)] for i in idx]


def auto_select_candidate(args, candidates):
    if len(candidates) <= 1:
        return candidates[0]
    image_root = join(args.path, args.image)
    scales = [float(x) for x in args.scales.split(",") if x.strip() != ""]
    params = create_detector_params(detect_inverted=args.detect_inverted)
    frame_filter = parse_frame_filter(args.frames)
    cam_filter = set(parse_csv_list(args.cams))

    cams = list_cameras(image_root)
    if len(cam_filter) > 0:
        cams = [c for c in cams if c in cam_filter]
    if len(cams) == 0:
        raise RuntimeError(f"No camera folder found in {image_root}")

    probe = {i: {"hits": 0, "total": 0, "frames": 0} for i in range(len(candidates))}
    for cam in cams:
        frame_map = collect_images(
            image_root,
            cam,
            args.ext,
            frame_filter=frame_filter,
            step=args.auto_probe_step,
        )
        frames = sample_probe_frames(frame_map, args.auto_probe_per_cam)
        for frame in frames:
            img = cv2.imread(frame_map[frame])
            if img is None:
                continue
            work, _ = prepare_detect_image(img, cam, args)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            for i, cand in enumerate(candidates):
                ids, _ = detect_charuco(
                    gray,
                    cand["dictionary"],
                    cand["board"],
                    params,
                    scales,
                    args.auto_probe_min_corners,
                )
                n = 0 if ids is None else int(ids.shape[0])
                probe[i]["total"] += n
                probe[i]["hits"] += int(n >= args.auto_probe_min_corners)
                probe[i]["frames"] += 1

    ranking = sorted(
        range(len(candidates)),
        key=lambda i: (probe[i]["hits"], probe[i]["total"]),
        reverse=True,
    )
    print("[auto] candidate ranking (top 8):")
    for i in ranking[:8]:
        c = candidates[i]
        s = probe[i]
        print(
            f"  {c['name']}: hits={s['hits']}, total_corners={s['total']}, probe_frames={s['frames']}"
        )
    best = ranking[0]
    if probe[best]["total"] <= 0:
        print("[auto] warning: all candidates have zero detected corners on probe set")
    return candidates[best]


def auto_select_aruco_candidate(args, candidates):
    if len(candidates) <= 1:
        return candidates[0]
    image_root = join(args.path, args.image)
    scales = [float(x) for x in args.scales.split(",") if x.strip() != ""]
    params = create_detector_params(detect_inverted=args.detect_inverted)
    frame_filter = parse_frame_filter(args.frames)
    cam_filter = set(parse_csv_list(args.cams))

    cams = list_cameras(image_root)
    if len(cam_filter) > 0:
        cams = [c for c in cams if c in cam_filter]
    if len(cams) == 0:
        raise RuntimeError(f"No camera folder found in {image_root}")

    probe = {i: {"hits": 0, "total": 0, "frames": 0} for i in range(len(candidates))}
    for cam in cams:
        frame_map = collect_images(
            image_root,
            cam,
            args.ext,
            frame_filter=frame_filter,
            step=args.auto_probe_step,
        )
        frames = sample_probe_frames(frame_map, args.auto_probe_per_cam)
        for frame in frames:
            img = cv2.imread(frame_map[frame])
            if img is None:
                continue
            work, _ = prepare_detect_image(img, cam, args)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            for i, cand in enumerate(candidates):
                _, marker_ids, obj, _ = detect_aruco_points(
                    gray,
                    cand["dictionary"],
                    cand["marker_lookup"],
                    params,
                    scales,
                )
                n = 0 if marker_ids is None else int(len(marker_ids))
                npts = 0 if obj is None else int(obj.shape[0])
                probe[i]["total"] += n
                probe[i]["hits"] += int(n >= args.auto_probe_min_markers)
                probe[i]["frames"] += 1
                probe[i]["total_pts"] = probe[i].get("total_pts", 0) + npts

    ranking = sorted(
        range(len(candidates)),
        key=lambda i: (probe[i]["hits"], probe[i]["total"], probe[i].get("total_pts", 0)),
        reverse=True,
    )
    print("[auto-aruco] candidate ranking (top 8):")
    for i in ranking[:8]:
        c = candidates[i]
        s = probe[i]
        print(
            f"  {c['name']}: hits={s['hits']}, total_markers={s['total']}, total_pts={s.get('total_pts', 0)}, probe_frames={s['frames']}"
        )
    best = ranking[0]
    if probe[best]["total"] <= 0:
        print("[auto-aruco] warning: all candidates have zero detected markers on probe set")
    return candidates[best]


def run_detection(args, candidates, corners3d):
    image_root = join(args.path, args.image)
    out_chess = join(args.out, "chessboard")
    out_vis = join(args.out, "output", "calibration")
    scales = [float(x) for x in args.scales.split(",") if x.strip() != ""]
    params = create_detector_params(detect_inverted=args.detect_inverted)
    frame_filter = parse_frame_filter(args.frames)
    cam_filter = set(parse_csv_list(args.cams))

    cams = list_cameras(image_root)
    if len(cam_filter) > 0:
        cams = [c for c in cams if c in cam_filter]
    if len(cams) == 0:
        raise RuntimeError(f"No camera folder found in {image_root}")

    detections = {cam: {} for cam in cams}
    image_sizes = {}
    dict_votes = {c["name"]: 0 for c in candidates}

    for cam in cams:
        frame_map = collect_images(
            image_root, cam, args.ext, frame_filter=frame_filter, step=args.step
        )
        for frame, imgname in frame_map.items():
            img = cv2.imread(imgname)
            if img is None:
                continue
            h, w = img.shape[:2]
            image_sizes[cam] = (w, h)
            work, tf = prepare_detect_image(img, cam, args)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            best_name = None
            best_ids, best_corners2d = None, None
            best_n = -1
            for c in candidates:
                ids, corners2d = detect_charuco(
                    gray,
                    c["dictionary"],
                    c["board"],
                    params,
                    scales,
                    args.min_corners,
                )
                n = 0 if ids is None else int(ids.shape[0])
                if n > best_n:
                    best_n = n
                    best_name = c["name"]
                    best_ids, best_corners2d = ids, corners2d
            if best_corners2d is not None:
                best_corners2d = map_corners_to_full_image(best_corners2d, tf)
            ids, corners2d = best_ids, best_corners2d
            if best_name is not None and best_n > 0:
                dict_votes[best_name] += 1
            detections[cam][frame] = {"ids": ids, "corners": corners2d}

            rec = to_chessboard_record(
                corners3d, ids, corners2d, args.long, args.short, args.square_length
            )
            save_json(join(out_chess, cam, f"{frame}.json"), rec)

            if args.write_vis:
                vis = img.copy()
                if tf["roi"] is not None:
                    x1, y1, x2, y2 = tf["roi"]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                if ids is not None and corners2d is not None:
                    draw_ids = ids.reshape(-1, 1)
                    draw_corners = corners2d.reshape(-1, 1, 2)
                    cv2.aruco.drawDetectedCornersCharuco(
                        vis, draw_corners, draw_ids, (0, 0, 255)
                    )
                os.makedirs(join(out_vis, cam), exist_ok=True)
                cv2.imwrite(join(out_vis, cam, f"{frame}.jpg"), vis)

    print("[detect] done")
    for cam in cams:
        frames = sorted(detections[cam].keys())
        valid = 0
        best = (None, -1)
        for frame in frames:
            ids = detections[cam][frame]["ids"]
            n = 0 if ids is None else int(ids.shape[0])
            if n >= args.min_corners:
                valid += 1
            if n > best[1]:
                best = (frame, n)
        print(
            f"[detect] {cam}: total={len(frames)}, valid(>={args.min_corners})={valid}, best={best[0]}({best[1]})"
        )
    print("[detect] dictionary votes:")
    for name in sorted(dict_votes.keys()):
        print(f"  {name}: {dict_votes[name]}")
    return detections, image_sizes


def run_aruco_detection(args, candidates):
    if len(candidates) == 0:
        raise RuntimeError("No ArUco candidate available")
    if len(candidates) > 1:
        print("[aruco] warning: multiple candidates provided; using first one")
    cand = candidates[0]
    image_root = join(args.path, args.image)
    out_aruco = join(args.out, "aruco")
    out_vis = join(args.out, "output", "aruco")
    scales = [float(x) for x in args.scales.split(",") if x.strip() != ""]
    params = create_detector_params(detect_inverted=args.detect_inverted)
    frame_filter = parse_frame_filter(args.frames)
    cam_filter = set(parse_csv_list(args.cams))

    cams = list_cameras(image_root)
    if len(cam_filter) > 0:
        cams = [c for c in cams if c in cam_filter]
    if len(cams) == 0:
        raise RuntimeError(f"No camera folder found in {image_root}")

    detections = {cam: {} for cam in cams}
    image_sizes = {}
    valid_frames = 0

    for cam in cams:
        frame_map = collect_images(
            image_root, cam, args.ext, frame_filter=frame_filter, step=args.step
        )
        for frame, imgname in frame_map.items():
            img = cv2.imread(imgname)
            if img is None:
                continue
            h, w = img.shape[:2]
            image_sizes[cam] = (w, h)
            work, tf = prepare_detect_image(img, cam, args)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

            marker_corners, marker_ids, obj, imgp = detect_aruco_points(
                gray,
                cand["dictionary"],
                cand["marker_lookup"],
                params,
                scales,
            )

            full_marker_corners = map_marker_corners_to_full_image(marker_corners, tf)
            if imgp is not None:
                imgp = map_corners_to_full_image(imgp, tf)

            nmarkers_raw = 0 if marker_ids is None else int(len(marker_ids))
            npts = 0 if imgp is None else int(imgp.shape[0])
            # Use matched markers (after ID-layout filtering) for calibration gating.
            nmarkers = int(npts // 4)
            if nmarkers >= args.aruco_min_markers:
                valid_frames += 1

            rec = {
                "ids": [] if marker_ids is None else marker_ids.reshape(-1).astype(int).tolist(),
                "nmarkers": nmarkers,
                "nmarkers_raw": nmarkers_raw,
                "npoints": npts,
                "image_path": imgname,
            }
            save_json(join(out_aruco, cam, f"{frame}.json"), rec)

            detections[cam][frame] = {
                "ids": marker_ids,
                "marker_corners": full_marker_corners,
                "obj": obj,
                "img": imgp,
                "nmarkers": nmarkers,
                "npts": npts,
            }

            if args.write_vis:
                vis = img.copy()
                if tf["roi"] is not None:
                    x1, y1, x2, y2 = tf["roi"]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                if full_marker_corners is not None and marker_ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, full_marker_corners, marker_ids)
                os.makedirs(join(out_vis, cam), exist_ok=True)
                cv2.imwrite(join(out_vis, cam, f"{frame}.jpg"), vis)

    print(
        f"[aruco-detect] done with candidate {cand['name']} (aruco={cand['aruco_type']}, marker={cand['marker_length']:.6f})"
    )
    for cam in cams:
        frames = sorted(detections[cam].keys())
        valid = 0
        best = (None, -1)
        for frame in frames:
            n = int(detections[cam][frame]["nmarkers"])
            if n >= args.aruco_min_markers:
                valid += 1
            if n > best[1]:
                best = (frame, n)
        print(
            f"[aruco-detect] {cam}: total={len(frames)}, valid(>={args.aruco_min_markers} markers)={valid}, best={best[0]}({best[1]})"
        )
    print(f"[aruco-detect] total valid frames across cameras: {valid_frames}")
    return detections, image_sizes


def calibrate_intrinsics(cams, detections, image_sizes, corners3d, min_views, min_corners):
    intri = {}
    for cam in cams:
        objpoints = []
        imgpoints = []
        for frame in sorted(detections[cam].keys()):
            ids = detections[cam][frame]["ids"]
            corners = detections[cam][frame]["corners"]
            if ids is None or corners is None or ids.shape[0] < min_corners:
                continue
            objpoints.append(corners3d[ids].astype(np.float32))
            imgpoints.append(corners.astype(np.float32))
        if len(objpoints) < min_views:
            print(f"[intri] {cam}: skip, only {len(objpoints)} valid views")
            continue
        if cam not in image_sizes:
            print(f"[intri] {cam}: skip, no valid image loaded")
            continue

        size = image_sizes[cam]
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, size, None, None
        )
        errs = []
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            proj = proj.reshape(-1, 2)
            err = np.linalg.norm(proj - imgpoints[i], axis=1).mean()
            errs.append(err)
        print(
            f"[intri] {cam}: views={len(objpoints)}, rms={ret:.4f}, reproj={np.mean(errs):.4f}px"
        )
        intri[cam] = {"K": K, "dist": dist}
    return intri


def calibrate_intrinsics_aruco(cams, detections, image_sizes, min_views, min_markers):
    intri = {}
    for cam in cams:
        objpoints = []
        imgpoints = []
        for frame in sorted(detections[cam].keys()):
            det = detections[cam][frame]
            obj = det["obj"]
            img = det["img"]
            if obj is None or img is None:
                continue
            if det["nmarkers"] < min_markers:
                continue
            objpoints.append(obj.astype(np.float32))
            imgpoints.append(img.astype(np.float32))
        if len(objpoints) < min_views:
            print(f"[aruco-intri] {cam}: skip, only {len(objpoints)} valid views")
            continue
        if cam not in image_sizes:
            print(f"[aruco-intri] {cam}: skip, no valid image loaded")
            continue

        size = image_sizes[cam]
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, size, None, None
        )
        errs = []
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            proj = proj.reshape(-1, 2)
            err = np.linalg.norm(proj - imgpoints[i], axis=1).mean()
            errs.append(err)
        print(
            f"[aruco-intri] {cam}: views={len(objpoints)}, rms={ret:.4f}, reproj={np.mean(errs):.4f}px"
        )
        intri[cam] = {"K": K, "dist": dist}
    return intri


def select_extri_frame(cams, detections, min_corners, frame_name=None):
    valid_sets = []
    for cam in cams:
        valid = set()
        for frame, det in detections[cam].items():
            ids = det["ids"]
            if ids is not None and ids.shape[0] >= min_corners:
                valid.add(frame)
        valid_sets.append(valid)
    common = sorted(list(set.intersection(*valid_sets)))
    if len(common) == 0:
        return None
    if frame_name is not None:
        if frame_name in common:
            return frame_name
        return None
    best = None
    best_score = -1
    for frame in common:
        score = 0
        for cam in cams:
            score += detections[cam][frame]["ids"].shape[0]
        if score > best_score:
            best_score = score
            best = frame
    return best


def select_extri_frame_aruco(cams, detections, min_markers, frame_name=None):
    valid_sets = []
    for cam in cams:
        valid = set()
        for frame, det in detections[cam].items():
            if int(det["nmarkers"]) >= int(min_markers):
                valid.add(frame)
        valid_sets.append(valid)
    common = sorted(list(set.intersection(*valid_sets)))
    if len(common) == 0:
        return None
    if frame_name is not None:
        return frame_name if frame_name in common else None
    best = None
    best_score = -1
    for frame in common:
        score = 0
        for cam in cams:
            score += int(detections[cam][frame]["npts"])
        if score > best_score:
            best_score = score
            best = frame
    return best


def calibrate_extrinsics(cams, detections, intri, corners3d, min_corners, frame_name=None):
    frame = select_extri_frame(cams, detections, min_corners, frame_name=frame_name)
    if frame is None:
        raise RuntimeError("No common frame with enough Charuco corners across all cameras.")
    print(f"[extri] use frame: {frame}")

    extri = {}
    for cam in cams:
        ids = detections[cam][frame]["ids"]
        corners = detections[cam][frame]["corners"]
        if ids is None or corners is None or ids.shape[0] < min_corners:
            raise RuntimeError(f"Camera {cam} has insufficient corners in frame {frame}")
        obj = corners3d[ids].astype(np.float32)
        img = corners.astype(np.float32)
        K = intri[cam]["K"]
        dist = intri[cam]["dist"]
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError(f"solvePnP failed for {cam}")
        R = cv2.Rodrigues(rvec)[0]
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img, axis=1).mean()
        center = -R.T @ tvec
        print(f"[extri] {cam}: err={err:.4f}px, center={center.reshape(-1)}")
        extri[cam] = {"rvec": rvec, "R": R, "T": tvec}
    return extri, frame


def calibrate_extrinsics_aruco(cams, detections, intri, min_markers, frame_name=None):
    frame = select_extri_frame_aruco(cams, detections, min_markers, frame_name=frame_name)
    if frame is None:
        raise RuntimeError("No common frame with enough ArUco markers across all cameras.")
    print(f"[aruco-extri] use frame: {frame}")

    extri = {}
    for cam in cams:
        det = detections[cam][frame]
        obj = det["obj"]
        img = det["img"]
        if obj is None or img is None or int(det["nmarkers"]) < int(min_markers):
            raise RuntimeError(
                f"Camera {cam} has insufficient ArUco markers in frame {frame}"
            )
        K = intri[cam]["K"]
        dist = intri[cam]["dist"]
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError(f"solvePnP failed for {cam}")
        R = cv2.Rodrigues(rvec)[0]
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img, axis=1).mean()
        center = -R.T @ tvec
        print(f"[aruco-extri] {cam}: err={err:.4f}px, center={center.reshape(-1)}")
        extri[cam] = {"rvec": rvec, "R": R, "T": tvec}
    return extri, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root path")
    parser.add_argument("--image", type=str, default="images")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--ext", type=str, default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "detect",
            "intri",
            "extri",
            "all",
            "reconstruct",
            "aruco_detect",
            "aruco_intri",
            "aruco_extri",
            "aruco_all",
        ],
    )
    parser.add_argument("--intri", type=str, default=None, help="intri.yml path for extri mode")
    parser.add_argument("--frame", type=str, default=None, help="fixed frame name for extri")
    parser.add_argument("--frames", type=str, default=None, help="only process these frames, csv")
    parser.add_argument("--step", type=int, default=1, help="keep every N-th filtered frame")
    parser.add_argument("--cams", type=str, default=None, help="only process these cameras, csv")
    parser.add_argument(
        "--template",
        type=str,
        default="custom",
        choices=sorted(BOARD_TEMPLATES.keys()),
        help="board template preset",
    )
    parser.add_argument(
        "--id_layout_json",
        type=str,
        default=None,
        help="custom reconstructed id layout json",
    )

    parser.add_argument("--long", type=int, default=12, help="board squares on long side")
    parser.add_argument("--short", type=int, default=8, help="board squares on short side")
    parser.add_argument(
        "--square_length",
        type=float,
        default=1.0,
        help="square length; use 1.0 when real size is unknown",
    )
    parser.add_argument(
        "--marker_length",
        type=float,
        default=-1.0,
        help="marker length; <=0 enables automatic ratio search",
    )
    parser.add_argument(
        "--aruco_type",
        type=str,
        default="auto",
        choices=["auto"] + sorted(ARUCO_DICT.keys()),
    )

    parser.add_argument("--min_corners", type=int, default=4)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--scales", type=str, default="1,2,3")
    parser.add_argument("--roi_json", type=str, default=None, help="per-camera ROI json")
    parser.add_argument(
        "--roi_format",
        type=str,
        default="xyxy",
        choices=["xyxy", "xywh"],
        help="list ROI format in json",
    )
    parser.add_argument(
        "--roi_upscale",
        type=float,
        default=2.0,
        help="crop upscale factor used when ROI exists",
    )
    parser.add_argument(
        "--auto_marker_ratios",
        type=str,
        default="0.50,0.55,0.60,0.65,0.70",
        help="ratio list marker_length/square_length for auto search",
    )
    parser.add_argument("--auto_probe_step", type=int, default=50)
    parser.add_argument("--auto_probe_per_cam", type=int, default=16)
    parser.add_argument("--auto_probe_min_corners", type=int, default=1)
    parser.add_argument("--auto_probe_min_markers", type=int, default=1)
    parser.add_argument(
        "--disable_auto_select",
        action="store_true",
        help="disable automatic best-candidate selection",
    )
    parser.add_argument("--recon_cam", type=str, default=None, help="camera name used for layout reconstruction")
    parser.add_argument("--recon_out", type=str, default=None, help="output json path for reconstructed id layout")
    parser.add_argument("--recon_min_markers", type=int, default=2, help="min markers per frame in reconstruction")
    parser.add_argument("--aruco_min_markers", type=int, default=3)
    parser.add_argument("--write_vis", action="store_true", help="write detection visualizations")
    parser.add_argument(
        "--detect_inverted",
        action="store_true",
        help="enable inverted ArUco marker detection",
    )
    args = parser.parse_args()
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not found. Install opencv-contrib-python.")
    if args.step < 1:
        raise ValueError("--step must be >= 1")
    if args.roi_upscale <= 0:
        raise ValueError("--roi_upscale must be > 0")
    if args.template != "custom":
        preset = BOARD_TEMPLATES[args.template]
        args.long = int(preset["long"])
        args.short = int(preset["short"])
        args.square_length = float(preset["square_length"])
        args.marker_length = float(preset["marker_length"])
        args.aruco_type = str(preset["aruco_type"])
    print(
        "[template]",
        args.template,
        f"long={args.long}, short={args.short}, square={args.square_length}, marker={args.marker_length}, aruco={args.aruco_type}",
    )
    if args.square_length <= 0:
        raise ValueError("--square_length must be > 0")

    if args.out is None:
        args.out = join(args.path, "output_charuco")
    os.makedirs(args.out, exist_ok=True)
    args.roi_map = load_roi_map(args.roi_json)
    if len(args.roi_map) > 0:
        print(
            f"[roi] loaded {len(args.roi_map)} entries from {args.roi_json}, format={args.roi_format}, upscale={args.roi_upscale}"
        )

    if args.mode == "reconstruct":
        reconstruct_id_layout(args)
        return

    marker_ratios = parse_float_list(args.auto_marker_ratios)
    marker_ratios = [r for r in marker_ratios if 0.0 < r < 1.0]
    if args.marker_length <= 0 and len(marker_ratios) == 0:
        raise ValueError("No valid auto marker ratios in --auto_marker_ratios")

    layout_data = load_id_layout(args.id_layout_json)
    custom_ids = None
    if layout_data is not None:
        if "board_order_ids" in layout_data:
            custom_ids = layout_data["board_order_ids"]
        if "aruco_type" in layout_data and args.aruco_type == "auto":
            args.aruco_type = str(layout_data["aruco_type"])
        if "long" in layout_data and int(layout_data["long"]) != int(args.long):
            print(
                f"[layout] warning: layout long={layout_data['long']} != current long={args.long}"
            )
        if "short" in layout_data and int(layout_data["short"]) != int(args.short):
            print(
                f"[layout] warning: layout short={layout_data['short']} != current short={args.short}"
            )
        if custom_ids is not None:
            print(
                f"[layout] loaded custom ids from {args.id_layout_json}, count={len(custom_ids)}"
            )

    candidates, corners3d = create_board_candidates(
        args.aruco_type,
        args.long,
        args.short,
        args.square_length,
        args.marker_length,
        marker_ratios=marker_ratios,
        custom_ids=custom_ids,
    )
    print(f"[auto] generated {len(candidates)} board candidates")
    is_aruco_mode = args.mode.startswith("aruco")
    if not args.disable_auto_select and len(candidates) > 1:
        if is_aruco_mode:
            best = auto_select_aruco_candidate(args, candidates)
        else:
            best = auto_select_candidate(args, candidates)
        candidates = [best]
        corners3d = best["corners3d"]
        args.aruco_type = best["aruco_type"]
        args.marker_length = best["marker_length"]
        print(
            f"[auto] selected candidate: {best['name']} (aruco={args.aruco_type}, marker={args.marker_length:.6f}, square={args.square_length:.6f})"
        )

    if is_aruco_mode:
        detections, image_sizes = run_aruco_detection(args, candidates)
        cams = sorted(detections.keys())
        intri = None
        if args.mode in ["aruco_intri", "aruco_all"]:
            intri = calibrate_intrinsics_aruco(
                cams, detections, image_sizes, args.min_views, args.aruco_min_markers
            )
            write_intri(join(args.out, "intri.yml"), intri)
            print("[write] intri:", join(args.out, "intri.yml"))
        if args.mode in ["aruco_extri", "aruco_all"]:
            if intri is None:
                if args.intri is None:
                    args.intri = join(args.out, "intri.yml")
                intri = read_intri(args.intri)
            extri, frame = calibrate_extrinsics_aruco(
                cams, detections, intri, args.aruco_min_markers, frame_name=args.frame
            )
            write_intri(join(args.out, "intri.yml"), intri)
            write_extri(join(args.out, "extri.yml"), extri)
            print("[write] intri:", join(args.out, "intri.yml"))
            print("[write] extri:", join(args.out, "extri.yml"))
            print("[aruco-extri] selected frame:", frame)
        return

    detections, image_sizes = run_detection(args, candidates, corners3d)
    cams = sorted(detections.keys())

    intri = None
    if args.mode in ["intri", "all"]:
        intri = calibrate_intrinsics(
            cams, detections, image_sizes, corners3d, args.min_views, args.min_corners
        )
        write_intri(join(args.out, "intri.yml"), intri)
        print("[write] intri:", join(args.out, "intri.yml"))

    if args.mode in ["extri", "all"]:
        if intri is None:
            if args.intri is None:
                args.intri = join(args.out, "intri.yml")
            intri = read_intri(args.intri)
        extri, frame = calibrate_extrinsics(
            cams, detections, intri, corners3d, args.min_corners, frame_name=args.frame
        )
        write_intri(join(args.out, "intri.yml"), intri)
        write_extri(join(args.out, "extri.yml"), extri)
        print("[write] intri:", join(args.out, "intri.yml"))
        print("[write] extri:", join(args.out, "extri.yml"))
        print("[extri] selected frame:", frame)


if __name__ == "__main__":
    main()
