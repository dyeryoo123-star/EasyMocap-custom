import argparse
import os

import cv2
import numpy as np


def read_names(node):
    out = []
    for i in range(node.size()):
        s = node.at(i).string()
        if s == "":
            s = str(int(node.at(i).real()))
        out.append(s)
    return out


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names = read_names(fs.getNode("names"))
    data = {}
    for cam in names:
        data[cam] = {
            "K": fs.getNode(f"K_{cam}").mat(),
            "dist": fs.getNode(f"dist_{cam}").mat(),
        }
    fs.release()
    return data


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
    cams = sorted(intri.keys())
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for cam in cams:
            f.write(f'  - "{cam}"\n')
        for cam in cams:
            write_matrix(f, f"K_{cam}", intri[cam]["K"])
            write_matrix(f, f"dist_{cam}", intri[cam]["dist"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="full intri.yml")
    parser.add_argument("--patch", type=str, required=True, help="new intri.yml with subset cameras")
    parser.add_argument("--cams", type=str, required=True, help="camera list csv to replace")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    base = read_intri(args.base)
    patch = read_intri(args.patch)
    cams = [x.strip() for x in args.cams.split(",") if x.strip() != ""]
    for cam in cams:
        if cam not in patch:
            raise ValueError(f"{cam} not found in patch intri")
        base[cam] = patch[cam]
        print(f"[replace] {cam}")
    write_intri(args.out, base)
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
