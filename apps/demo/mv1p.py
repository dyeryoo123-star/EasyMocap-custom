'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py
'''
from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import simple_recon_person, Timer, projectN3
from easymocap.pipeline import smpl_from_keypoints3d2d
import os
import subprocess
from os.path import join
import numpy as np

def smooth_skeleton(kp3ds, win=5):
    """Temporal moving-average smoothing for 3D keypoints.

    Only smooth xyz channels and keep confidence untouched.
    """
    if win <= 1 or len(kp3ds) == 0:
        return kp3ds
    arr = np.asarray(kp3ds, dtype=np.float32)  # [T, J, C]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return kp3ds
    radius = int(win // 2)
    out = arr.copy()
    for t in range(arr.shape[0]):
        l = max(0, t - radius)
        r = min(arr.shape[0], t + radius + 1)
        seg = arr[l:r]
        # confidence-weighted smoothing when confidence channel exists
        if arr.shape[-1] > 3:
            w = np.clip(seg[..., 3:4], 0.0, 1.0)
            wsum = np.sum(w, axis=0, keepdims=False) + 1e-8
            out[t, :, :3] = np.sum(seg[..., :3] * w, axis=0) / wsum
        else:
            out[t, :, :3] = np.mean(seg[..., :3], axis=0)
    return [out[t] for t in range(out.shape[0])]

def compose_video(img_dir, video_path, fps=30):
    """Compose a directory of numbered JPG frames into an MP4 video."""
    if not os.path.isdir(img_dir):
        return
    pattern = join(img_dir, '%06d.jpg')
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ]
    print('Composing video: {}'.format(video_path))
    subprocess.run(cmd, check=True)

def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    return keypoints3d, kpts_repro

def mv1pmf_skel(dataset, check_repro=True, args=None):
    MIN_CONF_THRES = args.thres2d
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    kpts_repro = None
    min_views = getattr(args, 'min_views', 2)
    strict_conf = getattr(args, 'strict_conf', None)
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        # ---- Extra hallucination filter ----
        kp = annots['keypoints']  # (V, J, 3)
        if strict_conf is not None and strict_conf > 0:
            kp[..., 2] = np.where(kp[..., 2] < strict_conf, 0.0, kp[..., 2])
        # require each joint to be seen by >= min_views cameras; otherwise zero it
        if min_views > 2:
            valid_cnt = (kp[..., 2] > 0).sum(axis=0)  # (J,)
            drop_joints = valid_cnt < min_views
            kp[:, drop_joints, 2] = 0.0
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        if check_repro:
            keypoints3d, kpts_repro = check_repro_error(keypoints3d, kpts_repro, annots['keypoints'], P=dataset.Pall, MAX_REPRO_ERROR=args.MAX_REPRO_ERROR)
        # keypoints3d, kpts_repro = robust_triangulate(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
        kp3ds.append(keypoints3d)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis)
    # smooth the skeleton
    if args.smooth3d > 0:
        kp3ds = smooth_skeleton(kp3ds, args.smooth3d)
    for nf in tqdm(range(len(kp3ds)), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf+start)
    if args.vis_repro and not args.no_video:
        compose_video(join(args.out, 'repro'), join(args.out, 'repro.mp4'))

def mv1pmf_smpl(dataset, args, weight_pose=None, weight_shape=None):
    dataset.skel_path = args.skel
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    keypoints2d, bboxes = [], []
    dataset.no_img = True
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        keypoints2d.append(annots['keypoints'])
        bboxes.append(annots['bbox'])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d)
    bboxes = np.stack(bboxes)
    kp3ds = check_keypoints(kp3ds, 1)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose)
    # write out the results
    dataset.no_img = not (args.vis_smpl or args.vis_repro)
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)
        dataset.write_smpl(param, nf)
        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            dataset.write_vertices(write_data, nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            dataset.vis_smpl(
                vertices=vertices[0], faces=body_model.faces, images=images, nf=nf,
                sub_vis=args.sub_vis, add_back=True, bboxes=annots['bbox'])
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')
    # compose rendered images into video
    if args.vis_smpl and not args.no_video:
        compose_video(join(args.out, 'smpl'), join(args.out, 'smpl.mp4'))
    if args.vis_repro and not args.no_video:
        compose_video(join(args.out, 'repro_smpl'), join(args.out, 'repro_smpl.mp4'))

if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument('--no_video', action='store_true',
        help='Do not compose rendered image folders into mp4 videos.')
    args = parse_parser(parser)
    help="""
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.model, args.gender, args.body)
    print(help)
    skel_path = join(args.out, 'keypoints3d')
    subs_offset = None
    if args.sync_offset is not None:
        import json as _json
        assert os.path.exists(args.sync_offset), args.sync_offset
        with open(args.sync_offset) as _f:
            _lags_data = _json.load(_f)
        subs_offset = _lags_data.get('lags', _lags_data)
        print('[mv1p] loaded sync offsets from {}: {}'.format(args.sync_offset, subs_offset))
    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose,
        subs_offset=subs_offset)
    dataset.writer.save_origin = args.save_origin

    if args.skel or not os.path.exists(skel_path):
        mv1pmf_skel(dataset, check_repro=True, args=args)
    mv1pmf_smpl(dataset, args)
    
