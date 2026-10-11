"""Sanity-check mocap by reprojecting 3D skeleton back onto each cam's
image and visually comparing with the original 2D detection.
"""
import argparse, json, os
import cv2, numpy as np


BODY25_SKELETON = [
    (1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16), (15, 17), (16, 18),
    (11, 24), (11, 22), (22, 23), (14, 21), (14, 19), (19, 20),
]


def read_intri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real())) for i in range(nn.size())]
    intri = {c: {"K": fs.getNode(f"K_{c}").mat(),
                 "dist": fs.getNode(f"dist_{c}").mat()} for c in names}
    fs.release(); return intri


def read_extri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real())) for i in range(nn.size())]
    extri = {c: {"R": fs.getNode(f"Rot_{c}").mat(),
                 "T": fs.getNode(f"T_{c}").mat()} for c in names}
    fs.release(); return extri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--frame", default="000100")
    ap.add_argument("--out", default="/tmp/mocap_check.jpg")
    args = ap.parse_args()

    intri = read_intri(os.path.join(args.data_root, "intri.yml"))
    extri = read_extri(os.path.join(args.data_root, "extri.yml"))
    cams = sorted(intri.keys())

    # load 3D keypoints
    kp3d = json.load(open(os.path.join(args.output_root, "keypoints3d",
                                         f"{args.frame}.json")))
    if isinstance(kp3d, list):
        kp3d = kp3d[0]
    pts3d = np.array(kp3d["keypoints3d"], dtype=np.float32)
    # add confidence col if missing
    if pts3d.shape[1] == 3:
        pts3d = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    print(f"loaded 3D keypoints: {pts3d.shape}")
    print(f"XYZ centroid: {pts3d[:,:3].mean(0)}")

    tiles = []
    for cam in cams:
        img_path = os.path.join(args.data_root, "images", cam,
                                  f"{args.frame}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  {cam}: NO IMAGE")
            continue
        K = intri[cam]["K"]; D = intri[cam]["dist"]
        R = extri[cam]["R"]; T = extri[cam]["T"].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(pts3d[:, :3].astype(np.float32),
                                     rvec, T, K, D)
        proj = proj.reshape(-1, 2)
        # also load 2D detection for comparison
        det_path = os.path.join(args.output_root, "keypoints2d", cam,
                                  f"{args.frame}.json")
        det = None
        if os.path.exists(det_path):
            dd = json.load(open(det_path))
            if isinstance(dd, list) and len(dd) > 0:
                if "keypoints" in dd[0]:
                    det = np.array(dd[0]["keypoints"])
        # draw
        for (a, b) in BODY25_SKELETON:
            if a < len(proj) and b < len(proj):
                p1 = tuple(np.round(proj[a]).astype(int))
                p2 = tuple(np.round(proj[b]).astype(int))
                cv2.line(img, p1, p2, (0, 255, 0), 3)
        for j, (x, y) in enumerate(proj):
            cv2.circle(img, (int(round(x)), int(round(y))), 5,
                       (0, 255, 255), -1)
        # red dots for 2D detection
        if det is not None:
            for j, kp in enumerate(det):
                if len(kp) >= 3 and kp[2] > 0.3:
                    cv2.circle(img, (int(kp[0]), int(kp[1])), 4,
                               (0, 0, 255), 2)
        # downscale
        h, w = img.shape[:2]
        scale = 480 / w
        img = cv2.resize(img, (480, int(h * scale)))
        cv2.putText(img, cam[:3], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        tiles.append(img)
        print(f"  {cam[:3]}: 3D reproj range x=[{proj[:,0].min():.0f},"
              f"{proj[:,0].max():.0f}] y=[{proj[:,1].min():.0f},"
              f"{proj[:,1].max():.0f}]")

    # uniform height
    if tiles:
        target_h = min(t.shape[0] for t in tiles)
        tiles = [cv2.resize(t, (480, target_h)) for t in tiles]
        row1 = np.hstack(tiles[:4])
        row2 = np.hstack(tiles[4:])
        big = np.vstack([row1, row2])
        # legend
        cv2.putText(big, "yellow/green = 3D reproj   red = 2D detection",
                    (15, big.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)
        cv2.imwrite(args.out, big, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"\nwrote {args.out}, shape={big.shape}")


if __name__ == "__main__":
    main()
