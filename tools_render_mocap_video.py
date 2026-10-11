"""Render an overlay video: each frame is a 2x4 mosaic of 8 cams with the
3D skeleton reprojected onto each cam's image.
"""

import argparse, json, os, subprocess
from glob import glob
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
    return {c:{'K':fs.getNode(f'K_{c}').mat(),'dist':fs.getNode(f'dist_{c}').mat()} for c in names}


def read_extri(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nn = fs.getNode("names")
    names = [nn.at(i).string() or str(int(nn.at(i).real())) for i in range(nn.size())]
    return {c:{'R':fs.getNode(f'Rot_{c}').mat(),'T':fs.getNode(f'T_{c}').mat()} for c in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--out_video', required=True)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--tile_w', type=int, default=640)
    ap.add_argument('--tile_h', type=int, default=360)
    ap.add_argument('--n_frames', type=int, default=-1)
    args = ap.parse_args()

    intri = read_intri(os.path.join(args.data_root, 'intri.yml'))
    extri = read_extri(os.path.join(args.data_root, 'extri.yml'))
    cams = sorted(intri.keys())

    frames = sorted(glob(os.path.join(args.output_root, 'keypoints3d', '*.json')))
    if args.n_frames > 0:
        frames = frames[:args.n_frames]
    print(f'rendering {len(frames)} frames')

    # writer
    H = args.tile_h * 2; W = args.tile_w * 4
    tmp_dir = '/tmp/mocap_render_anchored'
    os.makedirs(tmp_dir, exist_ok=True)
    for fi, kp_path in enumerate(frames):
        frame_id = os.path.splitext(os.path.basename(kp_path))[0]
        d = json.load(open(kp_path))
        if isinstance(d, list):
            d = d[0]
        pts3d = np.array(d['keypoints3d'], dtype=np.float32)
        if pts3d.shape[1] == 3:
            pts3d = np.hstack([pts3d, np.ones((pts3d.shape[0],1))])
        tiles = []
        for cam in cams:
            img_path = os.path.join(args.data_root, 'images', cam, f'{frame_id}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((args.tile_h, args.tile_w, 3), np.uint8)
            K = intri[cam]['K']; D = intri[cam]['dist']
            R = extri[cam]['R']; T = extri[cam]['T'].reshape(3,1)
            rvec,_ = cv2.Rodrigues(R)
            proj,_ = cv2.projectPoints(pts3d[:,:3].astype(np.float32), rvec, T, K, D)
            proj = proj.reshape(-1,2)
            for (a,b) in BODY25_SKELETON:
                if a<len(proj) and b<len(proj):
                    p1 = tuple(np.round(proj[a]).astype(int))
                    p2 = tuple(np.round(proj[b]).astype(int))
                    if all(-2000<v<5000 for v in p1+p2):
                        cv2.line(img, p1, p2, (0,255,0), 4)
            for (x,y) in proj:
                if -2000<x<5000 and -2000<y<5000:
                    cv2.circle(img, (int(round(x)),int(round(y))), 6, (0,255,255), -1)
            h,w = img.shape[:2]
            img = cv2.resize(img, (args.tile_w, args.tile_h))
            cv2.putText(img, cam[:3], (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            tiles.append(img)
        row1 = np.hstack(tiles[:4])
        row2 = np.hstack(tiles[4:])
        big = np.vstack([row1, row2])
        cv2.putText(big, f'frame {frame_id}', (15, H-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imwrite(os.path.join(tmp_dir, f'{fi:06d}.jpg'), big, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if fi % 50 == 0:
            print(f'  rendered {fi}/{len(frames)}')

    # ffmpeg to video
    cmd = ['ffmpeg', '-y', '-framerate', str(args.fps),
           '-i', os.path.join(tmp_dir, '%06d.jpg'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-crf', '20', args.out_video]
    print('encoding video:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f'wrote {args.out_video}')


if __name__ == '__main__':
    main()
