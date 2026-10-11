import torch
import numpy as np
import os
import cv2
from os.path import join
import pickle

def check_modelpath(paths):
    if isinstance(paths, str):
        assert os.path.exists(paths), paths
        return paths
    elif isinstance(paths, list):
        for path in paths:
            if os.path.exists(path):
                print(f'Found model in {path}')
                break
        else:
            print(f'No model found in {paths}!')
            raise FileExistsError
        return path
    else:
        raise NotImplementedError

class BaseYOLOv5:
    def __init__(self, ckpt=None, model='yolov5m', name='object2d',
                 multiview=True, flip180=False, conf_thres=None) -> None:
        if ckpt is not None:
            ckpt = check_modelpath(ckpt)
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', ckpt)
        else:
            print('[{}] Not given ckpt, use default yolov5'.format(self.__class__.__name__))
            self.model = torch.hub.load('ultralytics/yolov5', model)
        if conf_thres is not None:
            self.model.conf = float(conf_thres)
        self.multiview = multiview
        self.name = name
        # For ceiling-mounted (upside-down) cameras: rotate input image 180°
        # before YOLO, then flip detected bboxes back. YOLO was trained on
        # upright people; without this it misses most detections.
        self.flip180 = bool(flip180)
    
    def dump(self, cachename, output):
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        with open(cachename, 'wb') as f:
            pickle.dump(output, f)
        return output
    
    def load(self, cachename):
        with open(cachename, 'rb') as f:
            output = pickle.load(f)
        return output

    def check_cache(self, imgname):
        basename = os.path.basename(imgname)
        imgext = '.' + basename.split('.')[-1]
        nv = imgname.split(os.sep)[-2]
        cachename = join(self.output, self.name, nv, basename.replace(imgext, '.npy'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            output = self.load(cachename)
            return True, output, cachename
        else:
            return False, None, cachename
    
    def check_image(self, img_or_name):
        if isinstance(img_or_name, str):
            images = cv2.imread(img_or_name)
        else:
            images = img_or_name
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        return images
    
    @torch.no_grad()
    def detect(self, image, imgname):
        flag, cache, cachename = self.check_cache(imgname)
        if flag:
            return cache
        image = self.check_image(imgname)
        H, W = image.shape[:2]
        if self.flip180:
            image_in = image[::-1, ::-1].copy()
        else:
            image_in = image
        results = self.model(image_in)
        arrays = np.array(results.pandas().xyxy[0])
        if self.flip180 and len(arrays) > 0:
            # Flip bbox coords back: x' = W - 1 - x, y' = H - 1 - y
            new_x1 = W - 1 - arrays[:, 2].astype(float)
            new_y1 = H - 1 - arrays[:, 3].astype(float)
            new_x2 = W - 1 - arrays[:, 0].astype(float)
            new_y2 = H - 1 - arrays[:, 1].astype(float)
            arrays[:, 0] = new_x1
            arrays[:, 1] = new_y1
            arrays[:, 2] = new_x2
            arrays[:, 3] = new_y2
        res = {
            'results': arrays,
            'image_shape': image.shape,
        }
        self.dump(cachename, res)
        return res
    
    @staticmethod
    def select_class(results, name):
        select = []
        for i, res in enumerate(results['results']):
            classname = res[6]
            if classname != name:
                continue
            box = res[:5]
            select.append(box)
        if len(select) == 0:
            return np.zeros((0, 5), dtype=np.float32), results
        select = np.stack(select)
        return select, results

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        # Naive: select the best
        idx = np.argsort(select[:, -1])[::-1]
        return select[idx[0:1]]

    def __call__(self, images, imgnames): # 这里好像默认是多视角了，需要继承一下单视角的
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            squeeze = True
        detects = {'bbox': [[] for _ in range(len(images))]}
        for nv in range(len(images)):
            res = self.detect(images[nv], imgnames[nv])            
            select, res = self.select_class(res, self.name)
            if len(select) == 0:
                select = np.zeros((0,5), dtype=np.float32)
            else:
                select = np.stack(select).astype(np.float32)
            # TODO: add track here
            select = self.select_bbox(select, res, imgnames[nv])
            detects['bbox'][nv] = select
        if squeeze:
            detects['bbox'] = detects['bbox'][0]
        return detects

class YoloWithTrack(BaseYOLOv5):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track_cache = {}

    @staticmethod
    def calculate_iou(bbox_pre, bbox_now):
        area_now = (bbox_now[:, 2] - bbox_now[:, 0])*(bbox_now[:, 3]-bbox_now[:, 1])
        area_pre = (bbox_pre[:, 2] - bbox_pre[:, 0])*(bbox_pre[:, 3]-bbox_pre[:, 1])
        # compute IOU
        # max of left
        xx1 = np.maximum(bbox_now[:, 0], bbox_pre[:, 0])
        yy1 = np.maximum(bbox_now[:, 1], bbox_pre[:, 1])
        # min of right
        xx2 = np.minimum(bbox_now[:, 0+2], bbox_pre[:, 0+2])
        yy2 = np.minimum(bbox_now[:, 1+2], bbox_pre[:, 1+2])
        # w h
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        over = (w*h)/(area_pre+area_now-w*h)
        return over

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        sub = os.path.basename(os.path.dirname(imgname))
        frame = int(os.path.basename(imgname).split('.')[0])
        if sub not in self.track_cache:
            # select the best
            select = super().select_bbox(select, results, imgname)
            self.track_cache[sub] = {
                'frame': [frame],
                'bbox': [select]
            }
            return select
        bbox_pre = self.track_cache[sub]['bbox'][-1]
        iou = self.calculate_iou(bbox_pre, select)
        idx = iou.argmax()
        select = select[idx:idx+1]
        self.track_cache[sub]['frame'].append(frame)
        self.track_cache[sub]['bbox'].append(select)
        return select

class YoloWithManualInitTrack(BaseYOLOv5):
    """Track one target person per view from a manually selected first bbox.

    ``init_indices`` uses confidence-sorted candidate indices for each camera,
    matching the preview sheets where #0 is the highest confidence detection.
    After initialization, the selector keeps the candidate with the largest IoU
    to the previous target bbox, with center-distance fallback for missed/low
    overlap cases.
    """

    def __init__(self, init_indices=None, min_iou=0.05, center_weight=0.15,
                 **kwargs):
        super().__init__(**kwargs)
        self.init_indices = init_indices or {}
        self.min_iou = float(min_iou)
        self.center_weight = float(center_weight)
        self.track_cache = {}
        print(f'[YoloWithManualInitTrack] init_indices={self.init_indices}, '
              f'min_iou={self.min_iou}, center_weight={self.center_weight}')

    @staticmethod
    def calculate_iou(bbox_pre, bbox_now):
        area_now = (bbox_now[:, 2] - bbox_now[:, 0])*(bbox_now[:, 3]-bbox_now[:, 1])
        area_pre = (bbox_pre[:, 2] - bbox_pre[:, 0])*(bbox_pre[:, 3]-bbox_pre[:, 1])
        xx1 = np.maximum(bbox_now[:, 0], bbox_pre[:, 0])
        yy1 = np.maximum(bbox_now[:, 1], bbox_pre[:, 1])
        xx2 = np.minimum(bbox_now[:, 2], bbox_pre[:, 2])
        yy2 = np.minimum(bbox_now[:, 3], bbox_pre[:, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        denom = np.maximum(area_pre + area_now - w*h, 1e-6)
        return (w*h)/denom

    @staticmethod
    def center_distance_score(bbox_pre, bbox_now):
        c_pre = np.stack([(bbox_pre[:, 0] + bbox_pre[:, 2]) / 2,
                          (bbox_pre[:, 1] + bbox_pre[:, 3]) / 2], axis=-1)
        c_now = np.stack([(bbox_now[:, 0] + bbox_now[:, 2]) / 2,
                          (bbox_now[:, 1] + bbox_now[:, 3]) / 2], axis=-1)
        wh_pre = np.stack([bbox_pre[:, 2] - bbox_pre[:, 0],
                           bbox_pre[:, 3] - bbox_pre[:, 1]], axis=-1)
        scale = np.maximum(np.linalg.norm(wh_pre, axis=-1, keepdims=True), 1e-6)
        dist = np.linalg.norm(c_now - c_pre, axis=-1) / scale[:, 0]
        return 1.0 / (1.0 + dist)

    def _sort_by_conf(self, select):
        idx = np.argsort(select[:, -1])[::-1]
        return select[idx]

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        select = self._sort_by_conf(select)
        sub = os.path.basename(os.path.dirname(imgname))
        frame = int(os.path.basename(imgname).split('.')[0])
        if sub not in self.track_cache:
            init_idx = int(self.init_indices.get(sub, 0))
            init_idx = max(0, min(init_idx, select.shape[0] - 1))
            chosen = select[init_idx:init_idx+1]
            self.track_cache[sub] = {
                'frame': [frame],
                'bbox': [chosen],
                'init_idx': init_idx,
            }
            return chosen

        bbox_pre = self.track_cache[sub]['bbox'][-1]
        iou = self.calculate_iou(bbox_pre, select)
        center = self.center_distance_score(bbox_pre, select)
        conf = select[:, -1]
        score = iou + self.center_weight * center + 0.05 * conf
        idx = int(score.argmax())
        # If all candidates have near-zero IoU, the center-distance fallback
        # still keeps tracking rather than jumping to the highest confidence.
        if iou[idx] < self.min_iou:
            idx = int((center + 0.05 * conf).argmax())
        chosen = select[idx:idx+1]
        self.track_cache[sub]['frame'].append(frame)
        self.track_cache[sub]['bbox'].append(chosen)
        return chosen

class YoloWithManualInitTrackAndROI(YoloWithManualInitTrack):
    """Manual tracker with optional per-camera ROI override.

    Some ceiling views change candidate ordering between frames. For those
    views, a spatial ROI is more reliable than a fixed initial candidate id.
    """

    def __init__(self, roi_by_cam=None, roi_center_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.roi_by_cam = roi_by_cam or {}
        self.roi_center_weight = float(roi_center_weight)
        print(f'[YoloWithManualInitTrackAndROI] roi_by_cam={self.roi_by_cam}')

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        sub = os.path.basename(os.path.dirname(imgname))
        if sub not in self.roi_by_cam:
            return super().select_bbox(select, results, imgname)

        select = self._sort_by_conf(select)
        roi = self.roi_by_cam[sub]
        x1, y1, x2, y2 = [float(v) for v in roi[:4]]
        cx = (select[:, 0] + select[:, 2]) / 2
        cy = (select[:, 1] + select[:, 3]) / 2
        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        if inside.any():
            area = np.maximum((select[:, 2] - select[:, 0]) *
                              (select[:, 3] - select[:, 1]), 1.0)
            rcx, rcy = (x1 + x2) / 2, (y1 + y2) / 2
            rscale = max(np.hypot(x2 - x1, y2 - y1), 1.0)
            dist = np.hypot(cx - rcx, cy - rcy) / rscale
            score = np.log(area) + 0.05 * select[:, -1] - self.roi_center_weight * dist
            score[~inside] = -np.inf
            chosen = select[int(np.argmax(score)):int(np.argmax(score)) + 1]
            frame = int(os.path.basename(imgname).split('.')[0])
            self.track_cache.setdefault(sub, {'frame': [], 'bbox': [], 'init_idx': -1})
            self.track_cache[sub]['frame'].append(frame)
            self.track_cache[sub]['bbox'].append(chosen)
            return chosen

        return super().select_bbox(select, results, imgname)

class YoloWith3DAnchor(BaseYOLOv5):
    """Pick the bbox whose center is closest to a fixed 3D anchor's image
    projection. Removes the standard "pick highest confidence" behavior.

    Use this when the scene contains multiple people and you want to track a
    specific 3D location (e.g. a seated subject in a chair) consistently
    across all views. The 3D anchor is auto-refined frame-to-frame from the
    cams' selected bbox centroids."""

    def __init__(self, intri, extri, anchor_3d, anchor_radius_px=400,
                 update_anchor=True, **kwargs):
        super().__init__(**kwargs)
        import cv2 as _cv2
        from easymocap.mytools.camera_utils import read_cameras as _rc
        from os.path import dirname, basename
        # Allow either {intri,extri} as filenames in same dir, or full paths
        # If they're filenames only, derive root from `intri` arg's dir
        if dirname(intri):
            root = dirname(intri)
            self._cams = _rc(root, basename(intri), basename(extri))
        else:
            raise ValueError(f'intri must be a full path; got {intri}')
        self._anchor_3d = np.array(anchor_3d, dtype=np.float32)
        self._anchor_radius_px = float(anchor_radius_px)
        self._update_anchor = bool(update_anchor)
        # per-view caches
        self._last_bbox_centers = {}
        print(f'[YoloWith3DAnchor] anchor={self._anchor_3d}, '
              f'radius={self._anchor_radius_px}px, update={self._update_anchor}')

    def _project_to_cam(self, cam):
        c = self._cams[cam]
        proj, _ = cv2.projectPoints(self._anchor_3d.reshape(1, 1, 3),
                                     c['Rvec'], c['T'], c['K'], c['dist'])
        return float(proj[0, 0, 0]), float(proj[0, 0, 1])

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        cam = imgname.split(os.sep)[-2]
        if cam not in self._cams:
            return super().select_bbox(select, results, imgname)
        px, py = self._project_to_cam(cam)
        cx = (select[:, 0] + select[:, 2]) / 2
        cy = (select[:, 1] + select[:, 3]) / 2
        dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        # First filter: bboxes within anchor radius
        within = np.where(dist <= self._anchor_radius_px)[0]
        if len(within) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        # Among those, prefer the LARGEST bbox (seated subject typically the
        # most prominent body; small/partial detections from background
        # people are penalised even if they sit slightly closer to anchor).
        widths  = select[:, 2] - select[:, 0]
        heights = select[:, 3] - select[:, 1]
        areas   = widths * heights
        # Score: log-area minus normalised distance penalty
        # (large bbox close to anchor wins; tiny bbox close to anchor loses)
        log_area = np.log(np.maximum(areas, 1.0))
        dist_norm = dist / self._anchor_radius_px  # 0..1
        score = log_area - 1.5 * dist_norm
        # Restrict to in-radius candidates
        best = within[int(np.argmax(score[within]))]
        self._last_bbox_centers[cam] = (cx[best], cy[best])
        return select[best:best + 1]

    def __call__(self, images, imgnames):
        out = super().__call__(images, imgnames)
        if not self._update_anchor:
            return out
        bbox_list = out['bbox']
        if not isinstance(bbox_list, list):
            return out
        if len(bbox_list) < 2:
            return out
        cams_used = []; centers = []
        for nv, bb in enumerate(bbox_list):
            if bb.shape[0] == 0:
                continue
            cam = imgnames[nv].split(os.sep)[-2]
            if cam in self._cams:
                cams_used.append(cam)
                cx = (bb[0, 0] + bb[0, 2]) / 2
                cy = (bb[0, 1] + bb[0, 3]) / 2
                centers.append((cx, cy))
        if len(cams_used) < 3:
            return out
        # Robust anchor refit: pair-wise triangulation + per-axis MEDIAN.
        # Less sensitive to one or two cams selecting the wrong bbox than
        # full-view DLT (which would skew the SVD solution).
        try:
            from easymocap.mytools.camera_utils import Undistort
            from itertools import combinations
            estimates = []
            for i, j in combinations(range(len(cams_used)), 2):
                cam_i, cam_j = cams_used[i], cams_used[j]
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                c_i = self._cams[cam_i]; c_j = self._cams[cam_j]
                pu_i = Undistort.points(
                    np.array([[cx_i, cy_i, 1.0]], dtype=np.float32),
                    c_i['K'], c_i['dist'])
                pu_j = Undistort.points(
                    np.array([[cx_j, cy_j, 1.0]], dtype=np.float32),
                    c_j['K'], c_j['dist'])
                ui, vi = pu_i[0, 0], pu_i[0, 1]
                uj, vj = pu_j[0, 0], pu_j[0, 1]
                Pi, Pj = c_i['P'], c_j['P']
                A = np.array([
                    ui * Pi[2] - Pi[0], vi * Pi[2] - Pi[1],
                    uj * Pj[2] - Pj[0], vj * Pj[2] - Pj[1],
                ])
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]; X = X[:3] / X[3]
                # reject behind-camera or extreme estimates
                if abs(X[0]) < 30 and abs(X[1]) < 30 and 0.1 < X[2] < 30:
                    estimates.append(X)
            if len(estimates) < 3:
                return out
            X_med = np.median(np.array(estimates), axis=0).astype(np.float32)
            # gentle smoothing toward median estimate
            alpha = 0.30
            self._anchor_3d = (alpha * X_med +
                                (1 - alpha) * self._anchor_3d)
        except Exception:
            pass
        return out


class MultiPerson(BaseYOLOv5):
    def __init__(self, min_length, max_length, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        print('[{}] Only keep the bbox in [{}, {}]'.format(self.__class__.__name__, min_length, max_length))

    def select_bbox(self, select, results, imgname):
        if select.shape[0] == 0:
            return select
        # 判断一下面积
        area = np.sqrt((select[:, 2] - select[:, 0])*(select[:, 3]-select[:, 1]))
        valid = (area > self.min_length) & (area < self.max_length)
        height, width, _ = results['image_shape']
        # set the limit of left and right
        valid = valid & (select[:, 2] > self.min_length * 1.5) & (select[:, 0] < width - self.min_length * 1.5)
        return select[valid]

class DetectToPelvis:
    def __init__(self, key) -> None:
        self.key = key
        self.multiview = True
    
    def __call__(self, **kwargs):
        key = self.key
        val = kwargs[key]
        ret = {'pelvis': []}
        for nv in range(len(val)):
            bbox = val[nv]
            center = np.stack([(bbox[:, 0] + bbox[:, 2])/2, (bbox[:, 1] + bbox[:, 3])/2, bbox[:, -1]], axis=-1)
            ret['pelvis'].append(center)
        return ret

class Yolo_model:
    def __init__(self, mode, yolo_ckpt, multiview, repo_or_dir = 'ultralytics/yolov5', source='github') -> None:
        yolo_ckpt = check_modelpath(yolo_ckpt)
        self.model = torch.hub.load(repo_or_dir, 'custom', yolo_ckpt, source=source)
        self.min_detect_thres = 0.3
        self.mode = mode # 'fullimg' # 'bboxcrop'
        self.output = 'output'
        self.name = 'yolo'
        self.multiview = multiview
    @torch.no_grad()
    def det_step(self, img_or_name, imgname, bbox=[]):

        basename = os.path.basename(imgname)
        if self.multiview:
            nv = imgname.split('/')[-2]
            cachename = join(self.output, self.name, nv, basename.replace('.jpg', '.pkl'))
        else:
            cachename = join(self.output, self.name, basename.replace('.jpg', '.pkl'))
        os.makedirs(os.path.dirname(cachename), exist_ok=True)
        if os.path.exists(cachename):
            with open(cachename, 'rb') as f:
                output = pickle.load(f)
            return output

        if isinstance(img_or_name,str):
            images = cv2.imread(img_or_name)
        else:
            images = img_or_name

        if self.mode == 'bboxcrop':
            bbox[0] = max(0,bbox[0])
            bbox[1] = max(0,bbox[1])
            crop = images[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),::-1]
        else:
            crop = images[:,:,::-1]
        # print("[yolo img shape] ",crop.shape)
        results = self.model(crop) #RGB images[:,:,::-1]
        # breakpoint()
        arrays = np.array(results.pandas().xyxy[0])
        bboxes = {
            'bbox':[],
            'bbox_handl':[],
            'bbox_handr':[],
            'pelvis':[],
            'pelvis_l':[],
            'pelvis_r':[]
        }

        for i, res in enumerate(arrays):
            classid = res[5]
            box = res[:5]
            if self.mode == 'bboxcrop':
                box[0]+=bbox[0]
                box[2]+=bbox[0]
                box[1]+=bbox[1]
                box[3]+=bbox[1]
            if False:
                vis = images.copy()
                cpimg = crop.copy()
                from easymocap.mytools.vis_base import plot_bbox
                plot_bbox(vis,box,0)
                plot_bbox(cpimg,res[:5],0)
                cv2.imshow('vis',vis)
                # cv2.waitKey(0)
                cv2.imshow('crop',cpimg)
                cv2.waitKey(0)
                breakpoint()
            if box[4] < self.min_detect_thres:
                continue
            if classid==0:
                bboxes['bbox'].append(box)
            elif classid==1:
                bboxes['bbox_handl'].append(box)
                bboxes['pelvis_l'].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[-1]])
            elif classid==2:
                bboxes['bbox_handr'].append(box)
                bboxes['pelvis_r'].append([(box[0]+box[2])/2,(box[1]+box[3])/2,box[-1]])
        if(len(bboxes['bbox_handl'])==0):
            # bboxes['bbox_handl'].append(np.zeros((0, 5)))
            # bboxes['pelvis_l'].append(np.zeros((0, 3)))
            bboxes['bbox_handl'].append(np.zeros((5)))
            bboxes['pelvis_l'].append(np.zeros((3)))
            
        if(len(bboxes['bbox_handr'])==0):
            # bboxes['bbox_handr'].append(np.zeros((0, 5)))
            # bboxes['pelvis_r'].append(np.zeros((0, 3)))
            bboxes['bbox_handr'].append(np.zeros((5)))
            bboxes['pelvis_r'].append(np.zeros((3)))
        if(len(bboxes['bbox'])==0):
            bboxes['bbox'].append(np.zeros((5)))
        bboxes['bbox'] = np.array(bboxes['bbox'])
        if isinstance(imgname,str):
            with open(cachename, 'wb') as f:
                pickle.dump(bboxes, f)
        return bboxes
    def __call__(self, images, imgname, bbox=[]):
        return self.det_step(images, imgname, bbox)


class Yolo_model_hand_mvmp(Yolo_model):
    @torch.no_grad()
    def __call__(self, bbox, images, imgnames):
        ret = {
            'pelvis_l':[],
            'pelvis_r':[],
            # 'pelvis':[],
            'bbox_handl':[],
            'bbox_handr':[],
        }
        for nv in range(len(images)):
            img = images[nv]
            imgname = imgnames[nv]
            if self.mode == 'bboxcrop':
                bboxes = {
                    'bbox':[],
                    'bbox_handl':[],
                    'bbox_handr':[],
                    'pelvis_l':[],
                    'pelvis_r':[]
                }
                for pid in range(len(bbox[nv])):
                    bboxes_ = self.det_step(img, imgname, bbox[nv][pid])
                    for key in bboxes.keys():
                        bboxes[key].append(bboxes_[key])
            else:
                bboxes = self.det_step(img, imgname)
            for k in ret.keys():
                ret[k].append(np.array(bboxes[k]))

        return ret
