"""Scale per-cam K to a target resolution.

Takes an existing intri.yml (calibrated at some reference resolution per cam,
e.g. all 1080p) and produces a new intri.yml where selected cams have their
K rescaled to a new resolution (e.g. 1440p). Distortion coefficients are
unchanged because they live in normalized image coordinates.

Usage:
  python tools_scale_intri.py \\
      --in_intri  /Users/dyeryoo/Desktop/SmallSpaceA_DingYu_1300_1400/intri_1080p.yml \\
      --out_intri /Users/dyeryoo/Desktop/SmallSpaceA_DingYu_1300_1400/intri.yml \\
      --scale D03_20260325:1.3333333,D04_20260325:1.3333333,D08_20260325:1.3333333

Equivalently:
  --scale D03_20260325:2560/1920,D04_20260325:2560/1920,D08_20260325:2560/1920
"""

import argparse
import os
import shutil
import time

import cv2
import numpy as np


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        s = names_node.at(i).string()
        if s == "":
            s = str(int(names_node.at(i).real()))
        names.append(s)
    intri = {}
    for cam in names:
        K = fs.getNode(f"K_{cam}").mat()
        dist = fs.getNode(f"dist_{cam}").mat()
        intri[cam] = {"K": K, "dist": dist}
    fs.release()
    return intri, names


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


def write_intri(path, intri, names_order):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        shutil.copy(path, path + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
    with open(path, "w") as f:
        f.write("%YAML:1.0\n---\n")
        f.write("names:\n")
        for cam in names_order:
            f.write(f'  - "{cam}"\n')
        for cam in names_order:
            write_matrix(f, f"K_{cam}", intri[cam]["K"])
            write_matrix(f, f"dist_{cam}", intri[cam]["dist"])


def parse_scale(token):
    """'1.3333' or '2560/1920' -> float"""
    if "/" in token:
        a, b = token.split("/")
        return float(a) / float(b)
    return float(token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_intri", required=True)
    ap.add_argument("--out_intri", required=True)
    ap.add_argument("--scale", required=True,
                    help="comma list of cam:scale (e.g. "
                         "'D03_20260325:1.3333,D04_20260325:2560/1920')")
    args = ap.parse_args()

    scales = {}
    for pair in args.scale.split(","):
        pair = pair.strip()
        if not pair:
            continue
        cam, sc = pair.split(":")
        scales[cam.strip()] = parse_scale(sc.strip())

    intri, names = read_intri(args.in_intri)
    print(f"loaded {len(names)} cams from {args.in_intri}")

    out = {}
    for cam in names:
        K = intri[cam]["K"].copy()
        dist = intri[cam]["dist"].copy()
        if cam in scales:
            s = scales[cam]
            K[0, 0] *= s
            K[1, 1] *= s
            K[0, 2] *= s
            K[1, 2] *= s
            print(f"  [scale {s:.4f}] {cam}: "
                  f"fx {intri[cam]['K'][0,0]:.1f}->{K[0,0]:.1f} "
                  f"cx {intri[cam]['K'][0,2]:.1f}->{K[0,2]:.1f}")
        else:
            print(f"  [keep      ] {cam}: fx={K[0,0]:.1f}, cx={K[0,2]:.1f}")
        out[cam] = {"K": K, "dist": dist}

    write_intri(args.out_intri, out, names)
    print(f"\nwrote {args.out_intri}")


if __name__ == "__main__":
    main()
