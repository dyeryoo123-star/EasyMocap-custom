import json
import numpy as np
import open3d as o3d
from pathlib import Path

VERT_JSON = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/output/triangulate_fitSMPL/vertices/000000.json")

def load_vertices(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # 你的格式是 list[0]["vertices"]
    if isinstance(data, list):
        item = data[0]
    else:
        item = data

    if "vertices" not in item:
        raise ValueError(f"No 'vertices' key found in {json_path}")

    vertices = np.array(item["vertices"], dtype=np.float32)
    return vertices

def main():
    if not VERT_JSON.exists():
        raise FileNotFoundError(f"Not found: {VERT_JSON}")

    vertices = load_vertices(VERT_JSON)
    print("vertices shape:", vertices.shape)
    print("min:", vertices.min(axis=0))
    print("max:", vertices.max(axis=0))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    pcd.paint_uniform_color([0.2, 0.6, 1.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="Vertices Point Cloud",
        width=1280,
        height=720
    )

if __name__ == "__main__":
    main()