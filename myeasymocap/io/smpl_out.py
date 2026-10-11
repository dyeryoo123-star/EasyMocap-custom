import json
import numpy as np
import open3d as o3d
from pathlib import Path
from easymocap.bodymodel.smpl import SMPLModel
from myeasymocap.io.model import SMPLLoader

# 1) 改成你的输出路径
SMPL_JSON = Path("/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/output/triangulate_fitSMPL/smpl/000000.json")

# 2) 改成你本地的 SMPL 模型路径
MODEL_PATH = "/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

# 3) regressor 路径；如果没有，SMPLLoader 会尝试下载 J_regressor_body25.npy
REGRESSOR_PATH = "/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/data/smplx/J_regressor_body25.npy"



def load_smpl_json(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return data


def to_np(x):
    arr = np.array(x, dtype=np.float32)
    return arr


def main():
    print(f"Loading SMPL json: {SMPL_JSON}")
    if not SMPL_JSON.exists():
        raise FileNotFoundError(f"Not found: {SMPL_JSON}")

    param = load_smpl_json(SMPL_JSON)

    print("Keys:", param.keys())

    params = {
        "Rh": to_np(param["Rh"]),
        "Th": to_np(param["Th"]),
        "poses": to_np(param["poses"]),
        "shapes": to_np(param["shapes"]),
    }

    print("Rh shape:", params["Rh"].shape)
    print("Th shape:", params["Th"].shape)
    print("poses shape:", params["poses"].shape)
    print("shapes shape:", params["shapes"].shape)

    print("model path exists:", Path(MODEL_PATH).exists())
    print("regressor path exists:", Path(REGRESSOR_PATH).exists())

    print("before SMPLModel")
    model = SMPLModel(
        model_path=MODEL_PATH,
        regressor_path=REGRESSOR_PATH,
        device="cpu"
    )
    print("after SMPLModel")

    print("before forward")
    out = model(params, return_verts=True, return_tensor=False)
    print("after forward")

    # 兼容不同返回格式
    if isinstance(out, dict):
        print("output keys:", out.keys())
        if "vertices" in out:
            vertices = out["vertices"][0]
        elif "verts" in out:
            vertices = out["verts"][0]
        else:
            raise ValueError(f"Unknown output keys: {out.keys()}")
    else:
        vertices = out[0]

    faces = model.faces

    print("vertices shape:", np.array(vertices).shape)
    print("faces shape:", np.array(faces).shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.9])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    print("before Open3D")
    o3d.visualization.draw_geometries(
        [mesh, axis],
        window_name="SMPL Mesh",
        width=1280,
        height=720
    )
    print("after Open3D")


if __name__ == "__main__":
    main()