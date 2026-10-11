import json
import time
import numpy as np
import open3d as o3d
from pathlib import Path

VERT_DIR = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/output/triangulate_fitSMPL/vertices")
FACES_NPY = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/myeasymocap/io/smpl_faces.npy")   # 你先准备好的 faces

def load_vertices(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    item = data[0] if isinstance(data, list) else data
    return np.array(item["vertices"], dtype=np.float32)

def main():
    files = sorted(VERT_DIR.glob("*.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"No vertex json files found in {VERT_DIR}")

    faces = np.load(FACES_NPY)

    # 第一帧初始化
    vertices0 = load_vertices(files[0])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices0.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.9])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SMPL 3D Animation", width=1280, height=720)
    vis.add_geometry(mesh)
    vis.add_geometry(axis)

    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)

    for f in files:
        vertices = load_vertices(f)

        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.compute_vertex_normals()

        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)   # 大约 30fps

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()