"""Hunt for moments in D08's hour-long video where the 5x3 chessboard is
visible WITHOUT occlusion.

Strategy:
  1. Sample D08 every N seconds across the full duration
  2. Run SB detector
  3. For each successful detection, run single-frame cv2.solvePnP using
     D08's intrinsics (from intri.yml) against the 5x3 model
  4. Keep frames where reprojection error < threshold (~1.5 px) AND
     bounding box aspect ratio matches 5:3 within tolerance
  5. Save thumbnails of the best candidates + report time stamps

Output:
  /tmp/d08_hunt/clean_t<...>.jpg   — annotated thumbnails of good frames
  /tmp/d08_hunt/report.csv         — wall-clock time | err | bbox info
"""

import argparse
import csv
import json
import os
import subprocess
from os.path import join

import cv2
import numpy as np


def read_intri(path, cam):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode(f"K_{cam}").mat()
    dist = fs.getNode(f"dist_{cam}").mat()
    fs.release()
    return K, dist


def build_corners3d(pattern, grid):
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            pts[idx, 0] = c * grid
            pts[idx, 1] = r * grid
            idx += 1
    return pts


def detect(gray, pattern):
    pat_alt = (pattern[1], pattern[0])
    for pat in (pattern, pat_alt):
        try:
            ok, c = cv2.findChessboardCorners(
                gray, pat,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                      cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_FAST_CHECK)
        except cv2.error:
            ok = False
        if ok and c is not None:
            return pat, c.reshape(-1, 2)
    for pat in (pattern, pat_alt):
        for flags in (cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY,
                      cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE):
            try:
                ok, c = cv2.findChessboardCornersSB(gray, pat, flags=flags)
            except cv2.error:
                ok = False
            if ok and c is not None:
                return pat, c.reshape(-1, 2)
    return None, None


def fmt(sec):
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sync_meta', required=True)
    ap.add_argument('--intri', required=True)
    ap.add_argument('--cam', default='D08_20260325')
    ap.add_argument('--pattern', default='5x3')
    ap.add_argument('--grid', type=float, default=0.19)
    ap.add_argument('--t_start', default='13:02:18')
    ap.add_argument('--t_end',   default='14:09:00')
    ap.add_argument('--step_sec', type=float, default=2.0)
    ap.add_argument('--err_thresh_px', type=float, default=1.5)
    ap.add_argument('--out_dir', default='/tmp/d08_hunt')
    args = ap.parse_args()

    pat0 = tuple(int(x) for x in args.pattern.split('x'))
    meta = json.load(open(args.sync_meta))
    info = meta['cameras'][args.cam]
    video = info['video']
    K, dist = read_intri(args.intri, args.cam)
    print(f'cam={args.cam}, video={video}')
    print(f'K fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}')

    os.makedirs(args.out_dir, exist_ok=True)
    # parse times
    def parse(s):
        p = s.split(':'); return int(p[0])*3600+int(p[1])*60+float(p[2])
    t0 = parse(args.t_start); t1 = parse(args.t_end)

    rows = []
    n_sampled = 0; n_detected = 0; n_clean = 0
    t = t0
    while t < t1:
        t_in_v = t - info['start_sec_since_midnight']
        if t_in_v < 0 or t_in_v > info['duration_sec']:
            t += args.step_sec; continue
        # extract frame via ffmpeg to memory
        cmd = ['ffmpeg','-y','-ss',f'{t_in_v:.3f}','-i',video,
               '-vframes','1','-f','image2pipe','-vcodec','mjpeg','pipe:1']
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not r.stdout:
            t += args.step_sec; continue
        arr = np.frombuffer(r.stdout, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        n_sampled += 1
        if img is None:
            t += args.step_sec; continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pat, corners = detect(gray, pat0)
        if corners is None:
            t += args.step_sec; continue
        n_detected += 1
        # bbox
        xy = corners.reshape(-1, 2)
        bbox = (xy[:,0].max()-xy[:,0].min(),
                xy[:,1].max()-xy[:,1].min())
        # per-frame PnP self-consistency test
        obj = build_corners3d(pat, args.grid)
        try:
            ok, rv, tv = cv2.solvePnP(obj.astype(np.float32),
                                       corners.astype(np.float32),
                                       K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok = False
        if not ok:
            t += args.step_sec; continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        err = float(np.linalg.norm(proj.reshape(-1,2) - corners, axis=1).mean())
        bbox_w = float(bbox[0]); bbox_h = float(bbox[1])
        # extra geometric check: aspect of 5x3 should be longer than wider
        long_side = max(bbox_w, bbox_h); short_side = max(min(bbox_w, bbox_h),1.0)
        aspect = long_side / short_side
        is_clean = (err < args.err_thresh_px) and (1.2 < aspect < 4.0)
        if is_clean:
            n_clean += 1
            # save thumbnail
            annot = img.copy()
            cv2.drawChessboardCorners(annot, pat, corners.reshape(-1,1,2), True)
            annot = cv2.resize(annot, (1280, 720))
            label = fmt(t).replace(':','_')
            cv2.putText(annot,
                f'{args.cam} t={fmt(t)}  err={err:.2f}px  bbox={bbox_w:.0f}x{bbox_h:.0f}  pat={pat}',
                (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imwrite(join(args.out_dir, f'clean_{label}_e{err:.1f}.jpg'),
                        annot, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rows.append((fmt(t), err, bbox_w, bbox_h, aspect, is_clean,
                     pat[0], pat[1], t_in_v))
        if n_sampled % 200 == 0:
            print(f'  ... sampled={n_sampled}, det={n_detected}, clean={n_clean}')
        t += args.step_sec

    # report
    rep = join(args.out_dir, 'report.csv')
    with open(rep, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wallclock','err_px','bbox_w','bbox_h','aspect',
                    'is_clean','pat_w','pat_h','t_in_video'])
        for row in rows:
            w.writerow(row)
    print(f'\nsampled={n_sampled}, detected={n_detected}, '
          f'CLEAN={n_clean} (err<{args.err_thresh_px})')
    print(f'report -> {rep}')
    print(f'thumbnails -> {args.out_dir}/clean_*.jpg')


if __name__ == '__main__':
    main()
