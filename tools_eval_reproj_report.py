import glob
import json
import os
import argparse

import numpy as np

from easymocap.mytools.camera_utils import read_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="calibration_data/mocap_06800_07300")
    parser.add_argument("--out", default="calibration_data/mocap_06800_07300/output_relax/keypoints3d")
    parser.add_argument("--cams", nargs="+", default=[])
    args = parser.parse_args()

    root = args.root
    out = args.out
    if len(args.cams) > 0:
        camnames = args.cams
    else:
        camnames = [
            "D01_20260401170555",
            "D02_20260401170555",
            "D03_20260401170555",
            "D04_20260401170555",
            "D05_20260401170555",
            "D06_20260401170555",
            "D07_20260401170555",
            "D08_20260401170555",
        ]

    cams = read_camera(os.path.join(root, "intri.yml"), os.path.join(root, "extri.yml"))
    ann = {c: sorted(glob.glob(os.path.join(root, "annots", c, "*.json"))) for c in camnames}
    kfs = sorted(glob.glob(os.path.join(out, "*.json")))

    per = {c: [] for c in camnames}
    per_valid = {c: 0 for c in camnames}
    per_total = {c: 0 for c in camnames}

    for i, kf in enumerate(kfs):
        d = json.load(open(kf))
        if not d:
            continue
        k3 = np.array(d[0]["keypoints3d"], dtype=float)
        X = np.hstack([k3[:, :3], np.ones((k3.shape[0], 1))])
        for c in camnames:
            if i >= len(ann[c]):
                continue
            ad = json.load(open(ann[c][i]))
            aa = ad.get("annots", ad)
            if not aa:
                continue
            k2 = np.array(aa[0].get("keypoints", []), dtype=float)
            if k2.size == 0:
                continue
            P = cams[c]["P"]
            proj = (P @ X.T).T
            z = proj[:, 2:3]
            uv = proj[:, :2] / np.maximum(z, 1e-9)
            m = (k3[:, 3] > 0) & (k2[:, 2] > 0) & (z[:, 0] > 1e-6)
            per_total[c] += len(m)
            if m.sum() == 0:
                continue
            per_valid[c] += int(m.sum())
            e = np.sqrt(((uv[m] - k2[m, :2]) ** 2).sum(axis=1))
            per[c].extend(e.tolist())

    print("camera,n,valid_ratio,mean,median,p90,p95,gt100_ratio,gt200_ratio,gt400_ratio")
    for c in camnames:
        vals = np.array(per[c], dtype=float)
        if vals.size == 0:
            print(f"{c},0,0,NA,NA,NA,NA,NA,NA,NA")
            continue
        vr = per_valid[c] / max(per_total[c], 1)
        print(
            ",".join(
                [
                    c,
                    str(vals.size),
                    f"{vr:.3f}",
                    f"{vals.mean():.3f}",
                    f"{np.median(vals):.3f}",
                    f"{np.percentile(vals, 90):.3f}",
                    f"{np.percentile(vals, 95):.3f}",
                    f"{(vals > 100).mean():.3f}",
                    f"{(vals > 200).mean():.3f}",
                    f"{(vals > 400).mean():.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
