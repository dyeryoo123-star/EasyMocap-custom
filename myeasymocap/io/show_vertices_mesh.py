import json
import numpy as np
import open3d as o3d
from pathlib import Path

VERT_JSON = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/output/triangulate_fitSMPL/vertices/000000.json")
FACES_NPY = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/myeasymocap/io/smpl_faces.npy")

def load_vertices(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    item = data[0] if isinstance(data, list) else data
    return np.array(item["vertices"], dtype=np.float32)

def main():
    vertices = load_vertices(VERT_JSON)
    faces = np.load(FACES_NPY)

    print("vertices shape:", vertices.shape)
    print("faces shape:", faces.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.9])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [mesh, axis],
        window_name="SMPL Mesh",
        width=1280,
        height=720
    )

if __name__ == "__main__":
    main()