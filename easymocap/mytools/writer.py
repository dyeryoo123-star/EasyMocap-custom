import os
from os.path import join
import numpy as np
import cv2
# from mytools import save_json, merge
# from ..mytools import merge, plot_bbox, plot_keypoints
# from mytools.file_utils import read_json, save_json, read_annot, read_smpl, write_smpl, get_bbox_from_pose
from .vis_base import plot_bbox, plot_keypoints, merge
from .file_utils import write_keypoints3d, write_vertices, write_smpl, mkout, mkdir

class FileWriter:
    """
        This class provides:
                      |  write  | vis
        - keypoints2d |    x    |  o
        - keypoints3d |    x    |  o
        - smpl        |    x    |  o
    """
    def __init__(self, output_path, config=None, basenames=[], cfg=None) -> None:
        self.out = output_path
        keys = ['keypoints3d', 'match', 'smpl', 'skel', 'repro', 'keypoints']
        output_dict = {key:join(self.out, key) for key in keys}
        self.output_dict = output_dict
        
        self.basenames = basenames
        if cfg is not None:
            print(cfg, file=open(join(output_path, 'exp.yml'), 'w'))
        self.save_origin = False
        self.config = config
        self._smpl_renderer = None
        self._smpl_renderer_failed = False

    @staticmethod
    def _put_frame_number(img, nf):
        """Draw frame number on the top-left corner of the image."""
        H = img.shape[0]
        font_scale = max(H / 1000, 0.5)
        thickness = max(int(H / 500), 1)
        text = 'Frame {:06d}'.format(nf)
        # Black outline for readability
        cv2.putText(img, text, (10, int(30 * font_scale) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        # White text
        cv2.putText(img, text, (10, int(30 * font_scale) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def write_keypoints2d(self, ):
        pass

    def vis_keypoints2d_mv(self, images, lDetections, outname=None,
        vis_id=True, use_limb_color=False, nf=None):
        mkout(outname)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                pid = det['id']
                if 'keypoints2d' in det.keys():
                    keypoints = det['keypoints2d']
                else:
                    keypoints = det['keypoints']
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=use_limb_color, lw=2)
            images_vis.append(img)
        if len(images_vis) > 1:
            images_vis = merge(images_vis, resize=not self.save_origin)
        else:
            images_vis = images_vis[0]
        if nf is not None:
            self._put_frame_number(images_vis, nf)
        if outname is not None:
            cv2.imwrite(outname, images_vis)
        return images_vis
    
    def write_keypoints3d(self, results, outname):
        write_keypoints3d(outname, results)

    def write_vertices(self, results, outname):
        write_vertices(outname, results)
    
    def vis_keypoints3d(self, result, outname):
        # visualize the repro of keypoints3d
        import ipdb; ipdb.set_trace()
    
    def vis_smpl(self, render_data, images, cameras, outname, add_back, nf=None, bboxes=None):
        mkout(outname)
        render_results = None
        if not self._smpl_renderer_failed:
            try:
                if self._smpl_renderer is None:
                    from ..visualize.renderer import Renderer
                    self._smpl_renderer = Renderer(height=1024, width=1024, faces=None)
                render_results = self._smpl_renderer.render(
                    render_data, cameras, images, add_back=add_back)
            except Exception as e:
                self._smpl_renderer = None
                self._smpl_renderer_failed = True
                print('[vis_smpl] Renderer unavailable ({}: {}). '
                      'Fallback to projected wireframe mesh.'.format(
                          e.__class__.__name__, e))
        if render_results is None:
            render_results = self._vis_smpl_fallback(render_data, images, cameras, add_back=add_back, bboxes=bboxes)
        image_vis = merge(render_results, resize=not self.save_origin)
        if nf is not None:
            self._put_frame_number(image_vis, nf)
        cv2.imwrite(outname, image_vis)
        return image_vis

    @staticmethod
    def _project_vertices(vertices, K, R, T):
        T = np.asarray(T).reshape(3, 1)
        vert_cam = (R @ vertices.T + T).T
        z = vert_cam[:, 2]
        proj = (K @ vert_cam.T).T
        eps = 1e-6
        uv = proj[:, :2] / np.maximum(proj[:, 2:3], eps)
        return uv, z

    @staticmethod
    def _mesh_edges_from_faces(faces):
        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = np.vstack([e01, e12, e20])
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    @staticmethod
    def _adjust_to_bbox(uv, z, bbox):
        if bbox is None:
            return uv
        if len(bbox) < 4:
            return uv
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        valid = z > 1e-6
        if valid.sum() < 10:
            return uv
        uvv = uv[valid]
        px1, py1 = uvv[:, 0].min(), uvv[:, 1].min()
        px2, py2 = uvv[:, 0].max(), uvv[:, 1].max()
        pw = max(px2 - px1, 1.0)
        ph = max(py2 - py1, 1.0)
        # Scale mesh projection to person bbox for robust visual alignment.
        s = min(bw / pw, bh / ph) * 0.9
        c_pred = np.array([(px1 + px2) * 0.5, (py1 + py2) * 0.5], dtype=np.float32)
        c_tgt = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        return (uv - c_pred[None, :]) * s + c_tgt[None, :]

    def _vis_smpl_fallback(self, render_data, images, cameras, add_back=True, bboxes=None):
        out_images = []
        for nv, image in enumerate(images):
            if add_back:
                canvas = image.copy()
            else:
                canvas = np.zeros_like(image)
            H, W = canvas.shape[:2]
            K = cameras['K'][nv]
            R = cameras['R'][nv]
            T = cameras['T'][nv]
            for _, data in render_data.items():
                vertices = data['vertices']
                faces = data['faces']
                uv, z = self._project_vertices(vertices, K, R, T)
                if bboxes is not None and nv < len(bboxes):
                    uv = self._adjust_to_bbox(uv, z, bboxes[nv])
                faces_i = np.asarray(faces, dtype=np.int32)
                tri_z = z[faces_i].mean(axis=1)
                valid = (z[faces_i] > 1e-6).all(axis=1)
                tri_uv = uv[faces_i]
                # Keep triangles near image range to avoid expensive useless draw calls.
                xmin = tri_uv[:, :, 0].min(axis=1)
                xmax = tri_uv[:, :, 0].max(axis=1)
                ymin = tri_uv[:, :, 1].min(axis=1)
                ymax = tri_uv[:, :, 1].max(axis=1)
                in_view = (xmax >= -20) & (xmin <= W + 20) & (ymax >= -20) & (ymin <= H + 20)
                keep = np.where(valid & in_view)[0]
                if keep.shape[0] == 0:
                    continue
                order = keep[np.argsort(tri_z[keep])[::-1]]  # far -> near
                overlay = canvas.copy()
                # Subsample faces slightly for speed while preserving shape.
                for idx in order[::2]:
                    pts = np.round(tri_uv[idx]).astype(np.int32)
                    cv2.fillConvexPoly(overlay, pts, (196, 188, 242), lineType=cv2.LINE_AA)
                canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)
                # Draw sparse edges to preserve body contour.
                edges = self._mesh_edges_from_faces(faces_i)
                for i, j in edges[::3]:
                    if z[i] <= 1e-6 or z[j] <= 1e-6:
                        continue
                    x1, y1 = int(round(uv[i, 0])), int(round(uv[i, 1]))
                    x2, y2 = int(round(uv[j, 0])), int(round(uv[j, 1]))
                    if ((x1 < 0 and x2 < 0) or (x1 >= W and x2 >= W) or
                            (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H)):
                        continue
                    cv2.line(canvas, (x1, y1), (x2, y2), (88, 92, 140), 1, lineType=cv2.LINE_AA)
            out_images.append(canvas)
        return out_images

    def _write_keypoints3d(self, results, nf=-1, base=None):
        os.makedirs(self.output_dict['keypoints3d'], exist_ok=True)
        if base is None:
            base = '{:06d}'.format(nf)
        savename = join(self.output_dict['keypoints3d'], '{}.json'.format(base))
        save_json(savename, results)
    
    def vis_detections(self, images, lDetections, nf, key='keypoints', to_img=True, vis_id=True):
        os.makedirs(self.output_dict[key], exist_ok=True)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                if key == 'match' and 'id_match' in det.keys():
                    pid = det['id_match']
                else:
                    pid = det['id']
                if key not in det.keys():
                    keypoints = det['keypoints']
                else:
                    keypoints = det[key]
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        image_vis = merge(images_vis, resize=not self.save_origin)
        if to_img:
            savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)
        return image_vis
    
    def write_smpl(self, results, outname):
        write_smpl(outname, results)

    def vis_keypoints3d(self, infos, nf, images, cameras, mode='repro'):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        # cameras: (K, R, T)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            K, R, T = cameras['K'][nv], cameras['R'][nv], cameras['T'][nv]
            P = K @ np.hstack([R, T])
            for info in infos:
                pid = info['id']
                keypoints3d = info['keypoints3d']
                # 重投影
                kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
                kcam = kcam[:, :2]/kcam[:, 2:]
                k2d = np.hstack((kcam, keypoints3d[:, -1:]))
                bbox = get_bbox_from_pose(k2d, img)
                plot_bbox(img, bbox, pid=pid, vis_id=pid)
                plot_keypoints(img, k2d, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        savename = join(out, '{:06d}.jpg'.format(nf))
        image_vis = merge(images_vis, resize=False)
        cv2.imwrite(savename, image_vis)
        return image_vis

    def _vis_smpl(self, render_data_, nf, images, cameras, mode='smpl', base=None, add_back=False, extra_mesh=[]):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        from visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None, extra_mesh=extra_mesh)
        if isinstance(render_data_, list): # different view have different data
            for nv, render_data in enumerate(render_data_):
                render_results = render.render(render_data, cameras, images)
                image_vis = merge(render_results, resize=not self.save_origin)
                savename = join(out, '{:06d}_{:02d}.jpg'.format(nf, nv))
                cv2.imwrite(savename, image_vis)
        else:
            render_results = render.render(render_data_, cameras, images, add_back=add_back)
            image_vis = merge(render_results, resize=not self.save_origin)
            if nf != -1:
                if base is None:
                    base = '{:06d}'.format(nf)
                savename = join(out, '{}.jpg'.format(base))
                cv2.imwrite(savename, image_vis)
            return image_vis
