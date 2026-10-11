"""Sanity-check intrinsic calibration quality.

For each cam:
  1. Print full K / dist (fx, fy, cx, cy, k1, k2, p1, p2, k3)
  2. Re-detect chessboards and plot corner coverage on the image plane
     (red dots = all detected corners). A trustworthy calibration needs
     corners spread across most of the frame, especially toward the
     edges where distortion is largest.
  3. Compute FoV from K.
  4. Undistort one frame and save side-by-side (before/after) so we can
     visually check that straight lines stay straight.
"""

import argparse
import os
import subprocess
from glob import glob

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


_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def detect(gray, pattern=(7, 4)):
    try:
        ok, c = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                  cv2.CALIB_CB_NORMALIZE_IMAGE |
                  cv2.CALIB_CB_FAST_CHECK)
    except cv2.error:
        return None
    if not ok:
        return None
    c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), _SUBPIX)
    return c.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default="/Users/dyeryoo/Desktop/incal")
    ap.add_argument("--intri", required=True)
    ap.add_argument("--cams", default="D01,D02,D03,D04,D05,D06,D07,D08")
    ap.add_argument("--out_dir", default="/tmp/intri_analysis")
    ap.add_argument("--sample_every", type=int, default=60,
                    help="frame-stride for re-detection (60 = 1s @ 60fps)")
    ap.add_argument("--pattern", default="7x4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pat = tuple(int(x) for x in args.pattern.split("x"))
    cams = [c.strip() for c in args.cams.split(",")]

    intri, _ = read_intri(args.intri)
    name_map = {c: next((n for n in intri if n.startswith(c + "_") or n == c),
                        None) for c in cams}

    rows = []
    for cam in cams:
        key = name_map[cam]
        if key is None:
            print(f"[{cam}] not in intri.yml; skip")
            continue
        K = intri[key]["K"]; D = intri[key]["dist"].reshape(-1)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        k1, k2, p1, p2 = D[0], D[1], D[2], D[3]
        k3 = D[4] if len(D) > 4 else 0.0

        # find video
        vids = sorted(glob(os.path.join(args.videos_dir, f"{cam}_*.mp4")))
        if not vids:
            print(f"[{cam}] no video; skip"); continue
        cap = cv2.VideoCapture(vids[0])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fov_h = 2 * np.degrees(np.arctan(W / (2 * fx)))
        fov_v = 2 * np.degrees(np.arctan(H / (2 * fy)))

        # coverage: re-detect chessboards every N frames, accumulate corners
        coverage = np.zeros((H, W), dtype=np.uint8)
        n_hit = 0; one_good = None
        f_idx = 0
        while f_idx < N:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, img = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            c = detect(gray, pat)
            if c is not None:
                for (x, y) in c:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < W and 0 <= yi < H:
                        cv2.circle(coverage, (xi, yi), 8, 255, -1)
                n_hit += 1
                if one_good is None:
                    one_good = img.copy()
            f_idx += args.sample_every
        cap.release()
        coverage_frac = (coverage > 0).mean()

        # save coverage overlay
        cov_color = cv2.applyColorMap(coverage, cv2.COLORMAP_HOT)
        if one_good is not None:
            blend = cv2.addWeighted(one_good, 0.55, cov_color, 0.45, 0)
        else:
            blend = cov_color
        cv2.putText(blend, f"{cam} cov={coverage_frac*100:.1f}% n={n_hit}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(args.out_dir, f"{cam}_coverage.jpg"),
                    blend, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # undistort sample
        if one_good is not None:
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D.reshape(1, -1),
                                                     (W, H), alpha=0.0)
            und = cv2.undistort(one_good, K, D.reshape(1, -1), None, newK)
            sxs = np.hstack([one_good, und])
            sxs = cv2.resize(sxs, (sxs.shape[1] // 2, sxs.shape[0] // 2))
            cv2.putText(sxs, f"{cam}  L: raw   R: undistorted",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.out_dir, f"{cam}_undist.jpg"),
                        sxs, [cv2.IMWRITE_JPEG_QUALITY, 82])

        rows.append({
            "cam": cam, "size": (W, H), "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3,
            "fov_h": fov_h, "fov_v": fov_v,
            "n_hit": n_hit, "coverage_frac": coverage_frac,
        })

    print()
    print(f'{"cam":<5}{"fx":>9}{"fy":>9}{"cx":>9}{"cy":>9}'
          f'{"k1":>9}{"k2":>9}{"p1":>9}{"p2":>9}'
          f'{"FoV_h":>9}{"FoV_v":>9}{"hits":>7}{"cov%":>8}')
    print("-" * 120)
    for r in rows:
        print(f'{r["cam"]:<5}{r["fx"]:>9.1f}{r["fy"]:>9.1f}'
              f'{r["cx"]:>9.1f}{r["cy"]:>9.1f}'
              f'{r["k1"]:>+9.3f}{r["k2"]:>+9.3f}'
              f'{r["p1"]:>+9.4f}{r["p2"]:>+9.4f}'
              f'{r["fov_h"]:>9.1f}{r["fov_v"]:>9.1f}'
              f'{r["n_hit"]:>7d}{r["coverage_frac"]*100:>8.1f}')

    # consistency check across cams (same model expected — same lens)
    fxs = np.array([r["fx"] for r in rows])
    k1s = np.array([r["k1"] for r in rows])
    print()
    print(f"fx  : mean={fxs.mean():.1f}, std={fxs.std():.1f}, "
          f"min={fxs.min():.1f}, max={fxs.max():.1f}, "
          f"spread={(fxs.max()-fxs.min())/fxs.mean()*100:.1f}%")
    print(f"k1  : mean={k1s.mean():+.3f}, std={k1s.std():.3f}, "
          f"min={k1s.min():+.3f}, max={k1s.max():+.3f}")
    print(f"\noutputs -> {args.out_dir}/*_coverage.jpg, *_undist.jpg")


if __name__ == "__main__":
    main()
