"""Plain (non-MoE) ViTPose wrapper for the EasyMocap pipeline.

Compatible with checkpoints from
  https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/torch/coco_25
which produce 25-keypoint heatmaps in a body25-compatible ordering.

Drop-in replacement for `myeasymocap.backbone.hrnet.myhrnet.MyHRNet` in the
mv1p pipeline yaml: change the `module:` line and point `ckpt:` at the
downloaded .pth file.
"""

import os
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn

from .vit_moe import Attention, DropPath, PatchEmbed, Mlp, Head, trunc_normal_
from ..basetopdown import BaseTopDownModelCache
from ..topdown_keypoints import BaseKeypoints, coco17tobody25


# ---------------------------------------------------------------------------
# Vanilla (non-MoE) ViT backbone matching easy_ViTPose checkpoint layout.
# ---------------------------------------------------------------------------

class PlainBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PlainViT(nn.Module):
    """Vanilla ViT for pose estimation. Matches easy_ViTPose key layout:
    backbone.{pos_embed, patch_embed.*, blocks.<i>.*, last_norm.*}."""

    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3,
                 embed_dim=384, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0.0, ratio=1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim,
                                       ratio=ratio)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            PlainBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, drop_path=dpr[i],
                       norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        # easy_ViTPose convention: add pos_embed split into cls (idx 0) + patches
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp


class PlainComposeViT(nn.Module):
    """Backbone + single keypoint head, matching easy_ViTPose checkpoint."""

    def __init__(self, embed_dim=384, depth=12, num_heads=12, out_channels=25,
                 img_size=(256, 192), patch_size=16):
        super().__init__()
        self.backbone = PlainViT(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=4., qkv_bias=True, drop_path_rate=0.0, ratio=1)
        self.keypoint_head = Head(
            in_channels=embed_dim,
            out_channels=out_channels,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
        )

    def forward(self, x):
        feats = self.backbone(x)
        heatmaps = self.keypoint_head(feats)
        return {'output': heatmaps}


# ---------------------------------------------------------------------------
# coco_25 -> body25 conversion. easy_ViTPose coco_25 layout differs from COCO
# 17. Specifically it's [0..16] COCO + [17..24] feet-style extensions.
# We treat the extras as feet (body25 indices 19..24).
# ---------------------------------------------------------------------------

def coco25tobody25(points2d):
    """points2d: (N, 25, 3) in easy_ViTPose coco_25 order.
    Returns (N, 25, 3) in OpenPose body25 order.

    easy_ViTPose coco_25 layout:
        0 nose, 1 L_eye, 2 R_eye, 3 L_ear, 4 R_ear,
        5 neck, 6 L_shoulder, 7 R_shoulder, 8 L_elbow, 9 R_elbow,
        10 L_wrist, 11 R_wrist,
        12 L_hip, 13 R_hip, 14 mid_hip,
        15 L_knee, 16 R_knee, 17 L_ankle, 18 R_ankle,
        19..24 feet (L big toe, L small toe, L heel, R big toe, R small toe, R heel)

    OpenPose body25 layout:
        0 nose, 1 neck, 2 R_shoulder, 3 R_elbow, 4 R_wrist,
        5 L_shoulder, 6 L_elbow, 7 L_wrist,
        8 mid_hip, 9 R_hip, 10 R_knee, 11 R_ankle,
        12 L_hip, 13 L_knee, 14 L_ankle,
        15 R_eye, 16 L_eye, 17 R_ear, 18 L_ear,
        19..24 feet (L big toe, L small toe, L heel, R big toe, R small toe, R heel)
    """
    # coco_25 index -> body25 index
    mapping = [
        0,   # nose
        16,  # L_eye
        15,  # R_eye
        18,  # L_ear
        17,  # R_ear
        1,   # neck
        5,   # L_shoulder
        2,   # R_shoulder
        6,   # L_elbow
        3,   # R_elbow
        7,   # L_wrist
        4,   # R_wrist
        12,  # L_hip
        9,   # R_hip
        8,   # mid_hip
        13,  # L_knee
        10,  # R_knee
        14,  # L_ankle
        11,  # R_ankle
        19, 20, 21, 22, 23, 24,  # feet (already in same order)
    ]
    out = np.zeros_like(points2d)
    for src, dst in enumerate(mapping):
        out[:, dst] = points2d[:, src]
    return out


# ---------------------------------------------------------------------------
# EasyMocap wrapper module
# ---------------------------------------------------------------------------

# Preset configurations matching easy_ViTPose torch checkpoints.
_VARIANT_CFG = {
    's': dict(embed_dim=384,  depth=12, num_heads=12),
    'b': dict(embed_dim=768,  depth=12, num_heads=12),
    'l': dict(embed_dim=1024, depth=24, num_heads=16),
    'h': dict(embed_dim=1280, depth=32, num_heads=16),
}


class MyViTPosePlain(BaseTopDownModelCache, BaseKeypoints):
    """Drop-in MyHRNet replacement using easy_ViTPose checkpoints (no MoE).

    Args:
        ckpt:    path to .pth file from easy_ViTPose (coco_25 variant).
        variant: one of {'s','b','l','h'}; must match the checkpoint size.
        single_person: keep only the first person per view.
        num_joints: heatmap channel count of the checkpoint (25 for coco_25).
    """

    def __init__(self, ckpt, variant='b', single_person=True,
                 num_joints=25, name='keypoints2d',
                 mask_outside_bbox=False, mask_pad_px=20,
                 mask_color=(127, 127, 127)):
        """
        Args (added):
            mask_outside_bbox: if True, blank out image pixels outside the
                provided bbox (with `mask_pad_px` of padding) before running
                ViTPose. This stops nearby people / scene structure from
                bleeding into the crop and pulling keypoints off the target.
            mask_pad_px: pixels of padding around the bbox to retain
                (default 20).
            mask_color: BGR color used for masked-out regions
                (default mid-gray to keep input-norm stable).
        """
        super().__init__(name=name, bbox_scale=1.25,
                         res_input=[192, 256])
        self.mask_outside_bbox = bool(mask_outside_bbox)
        self.mask_pad_px = int(mask_pad_px)
        self.mask_color = tuple(int(c) for c in mask_color)
        if variant not in _VARIANT_CFG:
            raise ValueError(f"variant must be one of {list(_VARIANT_CFG)}")
        self.single_person = single_person
        self.num_joints = num_joints
        cfg = _VARIANT_CFG[variant]
        model = PlainComposeViT(out_channels=num_joints, **cfg)
        if not os.path.exists(ckpt):
            url = ("https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
                   f"torch/coco_25/vitpose-{variant}-coco_25.pth")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            print(f"[MyViTPosePlain] downloading {url} -> {ckpt}")
            import urllib.request
            urllib.request.urlretrieve(url, ckpt)
        state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # easy_ViTPose ckpts use plain 'backbone.*' and 'keypoint_head.*' keys
        # which match our module names directly.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[MyViTPosePlain] missing keys (first 5): {missing[:5]} ...")
        if unexpected:
            print(f"[MyViTPosePlain] unexpected keys (first 5): {unexpected[:5]} ...")
        model.eval()
        self.model = model
        self.device = (torch.device('cuda')
                       if torch.cuda.is_available()
                       else torch.device('cpu'))
        self.model.to(self.device)

    def dump(self, cachename, output):
        # Cache the post-processed 2D keypoints (not the raw heatmaps) to
        # save disk and re-decode time, mirroring MyViT.
        kpts = self.get_max_preds(output['output'])
        kpts_ori = self.batch_affine_transform(kpts, output['inv_trans'])
        kpts = np.concatenate([kpts_ori, kpts[..., -1:]], axis=-1)
        out = {'keypoints': kpts}
        super().dump(cachename, out)
        return out

    def estimate_keypoints(self, bbox, images, imgnames):
        squeeze = False
        if not isinstance(images, list):
            images = [images]
            imgnames = [imgnames]
            bbox = [bbox]
            squeeze = True
        nViews = len(images)
        kpts_all = []
        for nv in range(nViews):
            _bbox = bbox[nv]
            if _bbox.shape[0] == 0:
                if self.single_person:
                    kpts = np.zeros((1, self.num_joints, 3))
                else:
                    kpts = np.zeros((_bbox.shape[0], self.num_joints, 3))
            else:
                img_in = images[nv]
                if self.mask_outside_bbox and _bbox.shape[0] > 0:
                    # Mask all pixels outside the bbox (with padding) so the
                    # ViTPose crop sees only the target person.
                    if isinstance(img_in, str):
                        img_in = cv2.imread(img_in)
                    img_masked = np.full_like(img_in, 0)
                    img_masked[:] = self.mask_color
                    pad = self.mask_pad_px
                    H, W = img_in.shape[:2]
                    # Union over all bboxes (typically 1 in single-person mode)
                    for b in _bbox:
                        x1 = max(0, int(b[0]) - pad)
                        y1 = max(0, int(b[1]) - pad)
                        x2 = min(W, int(b[2]) + pad)
                        y2 = min(H, int(b[3]) + pad)
                        img_masked[y1:y2, x1:x2] = img_in[y1:y2, x1:x2]
                    img_in = img_masked
                out = super().__call__(_bbox, img_in, imgnames[nv])
                kpts = out['params']['keypoints']
            # coco_25 -> body25 ordering
            if kpts.shape[-2] == 25:
                kpts = coco25tobody25(kpts)
            elif kpts.shape[-2] == 17:
                kpts = self.coco17tobody25(kpts)
            kpts_all.append(kpts)
        if self.single_person:
            kpts_all = [k[0] for k in kpts_all]
            kpts_all = np.stack(kpts_all)
        if squeeze:
            kpts_all = kpts_all[0]
        return {'keypoints': kpts_all}

    def __call__(self, bbox, images, imgnames):
        return self.estimate_keypoints(bbox, images, imgnames)
