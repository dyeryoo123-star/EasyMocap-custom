import json
import numpy as np
import open3d as o3d
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

def create_skeleton_geometry(xyz, conf, conf_thr=0.01):
    valid = conf > conf_thr
    pts = xyz[valid].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(pts) > 0:
        pcd.paint_uniform_color([1.0, 0.0, 0.0])

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

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if len(lines) > 0:
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 1.0, 0.0]]), (len(lines), 1))
        )

    # 加一个坐标轴，方便找目标
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    return pcd, line_set, axis, pts

def show_single_frame(frame_id=0):
    json_path = KEYPOINTS_DIR / f"{frame_id:06d}.json"
    xyz, conf = load_keypoints3d(json_path)
    pcd, line_set, axis, pts = create_skeleton_geometry(xyz, conf)

    print("frame:", frame_id)
    print("valid points:", len(pts))
    if len(pts) > 0:
        print("min:", pts.min(axis=0))
        print("max:", pts.max(axis=0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_id:06d}", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(axis)

    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    show_single_frame(0)