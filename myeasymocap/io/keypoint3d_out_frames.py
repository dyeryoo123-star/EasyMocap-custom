import json
import numpy as np
import open3d as o3d
import time
from pathlib import Path

KEYPOINTS_DIR = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/output/triangulate_fitSMPL/keypoints3d")

BODY25_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20),
    (14, 21),
    (11, 22), (22, 23),
    (11, 24),
]

def load_keypoints3d(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    item = data[0] if isinstance(data, list) else data
    kpts = np.array(item["keypoints3d"], dtype=np.float32)
    xyz = kpts[:, :3]
    conf = kpts[:, 3] if kpts.shape[1] >= 4 else np.ones((kpts.shape[0],), dtype=np.float32)
    return xyz, conf

def build_geometry(xyz, conf, conf_thr=0.01):
    valid = conf > conf_thr
    pts = xyz[valid].astype(np.float64)

    index_map = {}
    cnt = 0
    for i, ok in enumerate(valid):
        if ok:
            index_map[i] = cnt
            cnt += 1

    lines = []
    for i, j in BODY25_EDGES:
        if i in index_map and j in index_map:
            lines.append([index_map[i], index_map[j]])
    return pts, lines

files = sorted(KEYPOINTS_DIR.glob("*.json"))[:100]
assert len(files) > 0, "No json files found"

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Skeleton Animation", width=1280, height=720)

pcd = o3d.geometry.PointCloud()
line_set = o3d.geometry.LineSet()
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

# 用第一帧初始化
xyz0, conf0 = load_keypoints3d(files[0])
pts0, lines0 = build_geometry(xyz0, conf0)

pcd.points = o3d.utility.Vector3dVector(pts0)
pcd.paint_uniform_color([1.0, 0.0, 0.0])

line_set.points = o3d.utility.Vector3dVector(pts0)
line_set.lines = o3d.utility.Vector2iVector(lines0)
if len(lines0) > 0:
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 1.0, 0.0]]), (len(lines0), 1))
    )

vis.add_geometry(pcd)
vis.add_geometry(line_set)
vis.add_geometry(axis)

ctr = vis.get_view_control()
ctr.set_front([0.0, 0.0, -1.0])
ctr.set_up([0.0, -1.0, 0.0])
ctr.set_zoom(0.8)

for f in files:
    xyz, conf = load_keypoints3d(f)
    pts, lines = build_geometry(xyz, conf)

    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if len(lines) > 0:
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 1.0, 0.0]]), (len(lines), 1))
        )

    vis.update_geometry(pcd)
    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.05)

vis.run()
vis.destroy_window()