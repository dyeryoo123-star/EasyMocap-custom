"""Merge per-camera intri_*.yml files into a single EasyMocap-format
intri.yml.

Each input file is expected to contain exactly one camera (the K_<name>
and dist_<name> for one cam, plus a 'names' list). The merged output
preserves the input cam-name order.

Usage:
  python tools_merge_intri.py \\
      --inputs /tmp/intri_out/intri_D01.yml /tmp/intri_out/intri_D02.yml ... \\
      --rename D01_20260401170555:D01,D02_20260401170555:D02,... \\
      --out /Users/dyeryoo/Desktop/SmallSpaceA_DingYu_1300_1400/intri.yml
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--rename", default="",
                    help="comma list of old:new pairs (e.g. "
                         "D01_20260401170555:D01,...). Old names absent "
                         "are kept verbatim.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rename = {}
    for pair in args.rename.split(","):
        pair = pair.strip()
        if not pair:
            continue
        old, new = pair.split(":")
        rename[old.strip()] = new.strip()

    merged = {}
    order = []
    for path in args.inputs:
        intri, names = read_intri(path)
        for cam in names:
            new_cam = rename.get(cam, cam)
            if new_cam in merged:
                print(f"[warn] duplicate cam {new_cam} from {path} — overwriting")
            merged[new_cam] = intri[cam]
            if new_cam not in order:
                order.append(new_cam)
            print(f"  + {new_cam}: fx={intri[cam]['K'][0,0]:.1f}, "
                  f"cx={intri[cam]['K'][0,2]:.1f}, "
                  f"cy={intri[cam]['K'][1,2]:.1f}, "
                  f"dist={intri[cam]['dist'].reshape(-1)[:3]}")

    write_intri(args.out, merged, order)
    print(f"\nwrote {len(order)} cams to {args.out}")


if __name__ == "__main__":
    main()
