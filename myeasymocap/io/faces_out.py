import numpy as np
from easymocap.bodymodel.smpl import SMPLModel

MODEL_PATH = "/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/data/bodymodels/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
REGRESSOR_PATH = "/Users/dyeryoo/Desktop/3D_capture/EasyMocap-master/data/smplx/J_regressor_body25.npy"

print("before SMPLModel")
model = SMPLModel(
    model_path=MODEL_PATH,
    regressor_path=REGRESSOR_PATH,
    device="cpu"
)
print("after SMPLModel")

faces = np.asarray(model.faces, dtype=np.int32)
print("faces shape:", faces.shape)

np.save("smpl_faces.npy", faces)
print("saved to smpl_faces.npy")