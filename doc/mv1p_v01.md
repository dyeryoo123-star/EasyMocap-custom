# mv1p.py (v0.1 Legacy Pipeline) -- Multi-View Single-Person Motion Capture

## 1. 运行命令

```bash
# 需要设置 EGL 渲染后端 (无头服务器上 pyrender 依赖此环境变量)
export PYOPENGL_PLATFORM=egl

python3 apps/demo/mv1p.py ${data} \
    --out ${data}/output/smpl \
    --vis_det --vis_repro --vis_smpl \
    --undis \
    --sub_vis 1 2 3 4
```

> **注意**: `--vis_smpl` 依赖 pyrender 进行 OpenGL 渲染。在无头服务器 (无 display) 上必须设置 `PYOPENGL_PLATFORM=egl`，否则会报 `pyglet.display.xlib.NoSuchDisplayException: Cannot connect to "None"`。测试机器使用 EGL 后端。

### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` (位置参数) | str | -- | 数据根目录 |
| `--out` | str | `{path}/output` | 输出目录 |
| `--sub` | str[] | 自动发现 | 指定使用的相机子目录 |
| `--sub_vis` | str[] | `[]` (全部) | 用于可视化的相机子集 |
| `--body` | str | `body25` | 关键点格式 (`body25`/`body15`/`bodyhand`/...) |
| `--model` | str | `smpl` | 人体模型类型 (`smpl`/`smplh`/`smplx`/`manol`/`manor`) |
| `--gender` | str | `neutral` | 性别 (`neutral`/`male`/`female`) |
| `--undis` | flag | False | 是否进行畸变矫正 |
| `--vis_det` | flag | False | 可视化 2D 检测结果 |
| `--vis_repro` | flag | False | 可视化三角化后的重投影 |
| `--vis_smpl` | flag | False | 可视化 SMPL 模型渲染结果 |
| `--thres2d` | float | 0.3 | 2D 关键点置信度阈值 |
| `--start` | int | 0 | 起始帧 |
| `--end` | int | 100000 | 结束帧 |
| `--smooth3d` | int | 0 | 3D 关键点平滑窗口大小 (0=不平滑) |
| `--MAX_REPRO_ERROR` | int | 50 | 重投影误差阈值 (像素) |
| `--skel` | flag | False | 是否强制重新计算骨架 |
| `--write_smpl_full` | flag | False | 额外写出完整 pose 参数 |
| `--write_vertices` | flag | False | 额外写出模型顶点 |
| `--verbose` | flag | False | 打印详细优化日志 |

---

## 2. 输入数据格式

### 2.1 目录结构

```
${data}/
├── intri.yml              # 相机内参 (OpenCV YAML 格式)
├── extri.yml              # 相机外参 (OpenCV YAML 格式)
├── images/
│   ├── 1/                 # 相机 1 的图片
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── 2/                 # 相机 2 的图片
│   ├── 3/
│   └── 4/
└── annots/
    ├── 1/                 # 相机 1 的 2D 标注
    │   ├── 000000.json
    │   ├── 000001.json
    │   └── ...
    ├── 2/
    ├── 3/
    └── 4/
```

### 2.2 相机参数文件

**intri.yml** -- 内参 (OpenCV YAML 格式):
```yaml
%YAML:1.0
---
names:
  - "1"
  - "2"
  - "3"
  - "4"
K_1: !!opencv-matrix         # 3x3 内参矩阵
  rows: 3
  cols: 3
  dt: d
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_1: !!opencv-matrix      # 1x5 畸变系数
  rows: 1
  cols: 5
  dt: d
  data: [k1, k2, p1, p2, k3]
# K_2, dist_2, ... 同理
```

**extri.yml** -- 外参 (OpenCV YAML 格式):
```yaml
%YAML:1.0
---
names:
  - "1"
  - "2"
  - "3"
  - "4"
R_1: !!opencv-matrix         # 3x1 旋转向量 (Rodrigues)
  rows: 3
  cols: 1
  dt: d
  data: [rx, ry, rz]
Rot_1: !!opencv-matrix       # 3x3 旋转矩阵
  rows: 3
  cols: 3
  dt: d
  data: [...]
T_1: !!opencv-matrix         # 3x1 平移向量
  rows: 3
  cols: 1
  dt: d
  data: [tx, ty, tz]
# R_2, Rot_2, T_2, ... 同理
```

读取逻辑 (`easymocap/mytools/camera_utils.py:read_camera`):
- 使用 `cv2.FileStorage` 读取 OpenCV YAML
- 从 Rodrigues 向量 `R_{cam}` 还原旋转矩阵 `Rot_{cam}`
- 计算投影矩阵 `P = K @ [R | T]`，存入 `cameras[cam]['P']`

### 2.3 2D 标注文件 (annots)

每帧每个相机一个 JSON 文件，格式：

```json
{
    "filename": "images/1/000000.jpg",
    "height": 1080,
    "width": 1920,
    "annots": [
        {
            "personID": 0,
            "bbox": [x1, y1, x2, y2, confidence],   // 5 个值
            "keypoints": [                            // Nx3, N=25 for body25
                [x, y, confidence],
                ...
            ]
        }
    ]
}
```

读取逻辑 (`easymocap/mytools/file_utils.py:read_annot`):
- `personID` 被重命名为 `id`
- `keypoints` 转为 `np.array`，形状 `(nJoints, 3)`
- 负置信度被置零
- 按 `id` 排序

---

## 3. 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│  main (__main__)                                                    │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐ │
│  │ 1. 参数解析 + 数据集创建  │   │ 2. 阶段一: 三角化 (mv1pmf_skel)│ │
│  │    load_parser()         │──>│    遍历每帧:                    │ │
│  │    parse_parser()        │   │    - 读取多视角 2D 关键点        │ │
│  │    MV1PMF(...)           │   │    - SVD 三角化 → 3D 关键点     │ │
│  │                          │   │    - 检查重投影误差              │ │
│  └─────────────────────────┘   │    - 写出 keypoints3d JSON       │ │
│                                └────────────────┬────────────────┘ │
│                                                  │                  │
│                                ┌────────────────▼────────────────┐ │
│                                │ 3. 阶段二: SMPL 拟合            │ │
│                                │    (mv1pmf_smpl)                │ │
│                                │    - 加载 3D 关键点 + 2D 关键点  │ │
│                                │    - 加载 SMPL 模型             │ │
│                                │    - 优化 shape → pose → 2D    │ │
│                                │    - 写出 SMPL 参数 JSON        │ │
│                                │    - 渲染可视化                  │ │
│                                └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 入口 (`apps/demo/mv1p.py:__main__`)

**代码位置**: `apps/demo/mv1p.py:118-141`

```python
parser = load_parser()          # 定义命令行参数
parser.add_argument('--skel')   # 额外的 --skel 参数
args = parse_parser(parser)     # 解析参数, 自动发现相机子目录

# 创建多视角单人数据集
dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub,
    out=args.out, config=CONFIG[args.body], kpts_type=args.body,
    undis=args.undis, no_img=False, verbose=args.verbose)

# 阶段一: 三角化 (如果 keypoints3d 目录不存在或指定 --skel)
if args.skel or not os.path.exists(skel_path):
    mv1pmf_skel(dataset, check_repro=True, args=args)

# 阶段二: SMPL 拟合 (始终执行)
mv1pmf_smpl(dataset, args)
```

执行流程:
1. `parse_parser()` 自动发现 `images/` 下的子目录作为相机列表 (如 `['1','2','3','4']`)，保存实验配置到 `exp.yml`
2. 构造 `MV1PMF` 数据集对象，读取相机参数、建立图像/标注列表
3. 先执行三角化得到 3D 骨架，再执行 SMPL 拟合

---

### 3.2 数据集初始化 (`MV1PMF` / `MVBase`)

**代码位置**: `easymocap/dataset/mv1pmf.py:14-75`, `easymocap/dataset/base.py:372-637`

`MV1PMF` 继承自 `MVBase`，初始化时:

1. **扫描文件列表**: 对每个相机 `cam`，收集 `images/{cam}/` 和 `annots/{cam}/` 下的文件列表
2. **确定帧数**: `nFrames = min(各相机图片数量)`
3. **读取相机参数** (`MVBase.read_camera`):
   - 从 `intri.yml` + `extri.yml` 读取每个相机的 K, R, T, dist
   - 计算投影矩阵 `P = K @ [R | T]`，堆叠为 `Pall`，形状 `(nViews, 3, 4)`
   - 计算各相机对的基础矩阵 `Fall`
4. **创建 FileWriter**: 负责后续的结果写入和可视化

`__getitem__(index)` 获取第 `index` 帧数据时:
- 读取所有视角的图片和 2D 标注
- 如果 `undis=True`，对图片做 `cv2.undistort`，对关键点做 `Undistort.points`
- `select_person` 按 `pid=0` 过滤出单人标注
- 返回 `images` (list of ndarray) 和 `annots` (dict: `bbox` (nViews,5), `keypoints` (nViews,25,3))

---

### 3.3 阶段一: 三角化 (`mv1pmf_skel`)

**代码位置**: `apps/demo/mv1p.py:44-69`

对每一帧执行:

```
for nf in range(start, end):
    1. images, annots = dataset[nf]
       # annots['keypoints']: (nViews, 25, 3)  -- 各视角的 2D 关键点
       # annots['bbox']:      (nViews, 5)      -- 各视角的边界框

    2. check_keypoints(annots['keypoints'], min_conf=0.3)
       # 将置信度 < 0.3 的关键点置零

    3. keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
       # 三角化: 多视角 2D → 3D

    4. check_repro_error(keypoints3d, kpts_repro, annots['keypoints'], P, MAX_REPRO_ERROR=50)
       # 重投影误差 > 50 像素的关键点置零, 重新三角化

    5. kp3ds.append(keypoints3d)

    6. [可选] vis_det  → 保存到 {out}/detec/{:06d}.jpg
    7. [可选] vis_repro → 保存到 {out}/repro/{:06d}.jpg
```

最终:
- 如果 `smooth3d > 0`，对 3D 关键点序列进行滑动窗口平滑
- 遍历写出 `{out}/keypoints3d/{:06d}.json`
- 如果 `vis_repro`，用 ffmpeg 合成 `{out}/repro.mp4`

#### 3.3.1 三角化算法 (`simple_recon_person`)

**代码位置**: `easymocap/mytools/reconstruction.py:93-100`

```python
def simple_recon_person(keypoints_use, Puse):
    out = batch_triangulate(keypoints_use, Puse)  # SVD 三角化
    kpts_repro = projectN3(out, Puse)              # 重投影到各视角
    return out, kpts_repro
```

**`batch_triangulate`** (`reconstruction.py:52-90`):
- 输入: `keypoints_` `(nViews, nJoints, 3)`, `Pall` `(nViews, 3, 4)`
- 对每个关节点，构造超定方程组 `A x = 0` (DLT 方法):
  - `A[2i]   = conf * (u * P[2] - P[0])`
  - `A[2i+1] = conf * (v * P[2] - P[1])`
- SVD 分解取最小奇异值对应的解
- 仅对 **至少 2 个视角有有效检测** 的关节点求解
- 输出: `(nJoints, 4)` -- `[x, y, z, conf_mean]`

**`projectN3`** (`reconstruction.py:17-29`):
- 将 3D 点投影回每个视角: `kp2d = P @ [x,y,z,1]^T`，然后做齐次坐标归一化

#### 3.3.2 重投影误差检查 (`check_repro_error`)

**代码位置**: `apps/demo/mv1p.py:33-42`

- 计算每个视角每个关节点的重投影误差 (像素距离)
- 对误差超过 `MAX_REPRO_ERROR` (默认 50 像素) 的关键点，将其置信度设为 0
- 用更新后的 2D 关键点重新三角化

---

### 3.4 阶段二: SMPL 拟合 (`mv1pmf_smpl`)

**代码位置**: `apps/demo/mv1p.py:71-116`

#### 3.4.1 数据加载

```python
# 1. 重新遍历数据集，收集 2D 关键点和 bbox
for nf in range(start, end):
    images, annots = dataset[nf]    # no_img=True, 不读图片
    keypoints2d.append(annots['keypoints'])   # (nViews, 25, 3)
    bboxes.append(annots['bbox'])             # (nViews, 5)

# 2. 从三角化结果加载 3D 关键点
kp3ds = dataset.read_skeleton(start, end)     # (nFrames, 25, 4)

# 3. 堆叠为 numpy 数组
keypoints2d = np.stack(keypoints2d)  # (nFrames, nViews, 25, 3)
bboxes = np.stack(bboxes)            # (nFrames, nViews, 5)
```

#### 3.4.2 SMPL 模型加载

**代码位置**: `easymocap/smplmodel/body_param.py:32-64`

```python
body_model = load_model(gender='neutral', model_type='smpl')
```

- 从 `data/smplx/smpl/` 加载 SMPL 模型参数
- 加载关节回归器 `J_regressor_body25.npy` (将 SMPL 的 6890 个顶点回归到 Body25 的 25 个关节点)
- 模型放到 CUDA 设备上

#### 3.4.3 优化过程 (`smpl_from_keypoints3d2d`)

**代码位置**: `easymocap/pipeline/basic.py:55-78`

分三个阶段进行优化:

**阶段 A: Shape 优化 (`optimizeShape`)**

**代码位置**: `easymocap/pyfitting/optimize_simple.py:17-82`

- 目标: 用 3D 关键点的骨长约束拟合 `shapes` (10 维)
- 优化变量: `shapes` (1, 10)
- 损失函数:
  - `s3d`: 模型骨长与实测骨长的差异 (加权 1.0)
  - `reg_shapes`: shape 参数的 L2 正则 (加权 5e-3)
- 优化器: L-BFGS (strong_wolfe line search)
- 使用 `body15` 的前 1-14 段肢体 (不含 nose, neck)

**阶段 B: 3D Pose 优化 (`optimizePose3D`)**

**代码位置**: `easymocap/pyfitting/optimize_simple.py:305-349`

分两步:

1. **全局 RT 优化** (`OPT_R=True, OPT_T=True`):
   - 优化变量: `Rh` (nFrames, 3), `Th` (nFrames, 3)
   - 损失: 3D 关键点距离 + 平滑 + 正则

2. **Body Pose 优化** (`OPT_POSE=True`):
   - 优化变量: `Rh`, `Th`, `poses` (nFrames, 72)
   - 损失函数 (各权重来自 `load_weight_pose`):
     - `k3d` (1.0): 模型关节与三角化关节的 3D 距离
     - `smooth_body` (5.0): 相邻帧关节位置平滑
     - `smooth_poses` (1.0): 相邻帧 pose 参数平滑
     - `smooth_Rh` (1.0): 相邻帧全局旋转平滑
     - `reg_poses` (1e-3): pose 参数 L2 正则
     - `reg_poses_zero` (1e-2): 特定关节 pose 归零正则
   - 优化器: L-BFGS

**阶段 C: 2D Pose 优化 (`optimizePose2D`)**

**代码位置**: `easymocap/pyfitting/optimize_simple.py:351-401`

- 优化变量: `Rh`, `Th`, `poses`
- 额外损失:
  - `k2d` (1e-4): 模型关节投影到各视角与 2D 检测的距离 (在 bbox 归一化坐标下)
- 使用多视角 2D 关键点联合约束

**所有优化阶段的通用流程** (`_optimizeSMPL`, `optimize_simple.py:244-303`):
```
1. 准备参数 (deepcopy, 零化 root pose, 插值空白帧)
2. 前向计算: kpts_est = body_model(poses, shapes, Rh, Th)
3. 计算各项损失 → 加权求和
4. L-BFGS 优化 (FittingMonitor 控制收敛: ftol=1e-4)
5. 后处理 (插值空白帧, tensor → numpy)
```

#### 3.4.4 结果写出与可视化

```python
for nf in range(start, end):
    # 1. 写出 SMPL 参数
    dataset.write_smpl(param, nf)
    #    → {out}/smpl/{:06d}.json

    # 2. [可选] 写出完整 poses (--write_smpl_full)
    #    → {out}/smpl_full/{:06d}.json

    # 3. [可选] 写出顶点 (--write_vertices)
    #    → {out}/vertices/{:06d}.json

    # 4. [可选] SMPL 渲染可视化 (--vis_smpl)
    #    body_model(return_verts=True) → 6890 个顶点
    #    pyrender 渲染到各视角图片上
    #    → {out}/smpl/{:06d}.jpg

    # 5. [可选] SMPL 重投影可视化 (--vis_repro)
    #    body_model(return_verts=False) → 关节点
    #    projectN3 投影到各视角
    #    → {out}/repro_smpl/{:06d}.jpg
```

---

## 4. 输出结果格式

### 4.1 输出目录结构

```
${out}/
├── exp.yml                # 实验参数快照
├── keypoints3d/           # 三角化的 3D 关键点
│   ├── 000000.json
│   ├── 000001.json
│   └── ...
├── smpl/                  # SMPL 参数
│   ├── 000000.json
│   ├── 000001.json
│   └── ...
├── detec/                 # [--vis_det] 2D 检测可视化
│   ├── 000000.jpg         # 多视角拼接图
│   └── ...
├── repro/                 # [--vis_repro] 三角化重投影可视化
│   ├── 000000.jpg
│   └── ...
├── repro.mp4              # [--vis_repro] 重投影可视化视频
├── smpl/                  # [--vis_smpl] SMPL 渲染可视化 (与参数同目录)
│   ├── 000000.jpg
│   └── ...
├── smpl.mp4               # [--vis_smpl] SMPL 渲染视频
├── repro_smpl/            # [--vis_repro + smpl 阶段] SMPL 关节重投影
│   ├── 000000.jpg
│   └── ...
└── repro_smpl.mp4         # SMPL 关节重投影视频
```

### 4.2 keypoints3d JSON 格式

每帧一个文件 (`{out}/keypoints3d/{:06d}.json`):

```json
[
    {
        "id": 0,
        "keypoints3d": [
            [x, y, z, confidence],   // 关节 0 (Nose)
            [x, y, z, confidence],   // 关节 1 (Neck)
            ...                      // 共 25 个关节 (Body25)
        ]
    }
]
```

- 形状: `(25, 4)` -- 每个关节 `[x, y, z, conf]`
- `conf` 为各视角置信度的加权平均值
- 坐标系: 世界坐标系 (由相机外参定义)

### 4.3 SMPL 参数 JSON 格式

每帧一个文件 (`{out}/smpl/{:06d}.json`):

```json
[
    {
        "id": 0,
        "Rh": [[rx, ry, rz]],         // (1, 3) 全局旋转 (axis-angle)
        "Th": [[tx, ty, tz]],         // (1, 3) 全局平移
        "poses": [[p0, p1, ..., p71]],  // (1, 72) 身体姿态参数 (24个关节 x 3, axis-angle)
        "shapes": [[s0, s1, ..., s9]]   // (1, 10) 体型参数 (PCA 系数)
    }
]
```

参数说明:
- `Rh`: 全局旋转，axis-angle 表示，3 维向量
- `Th`: 全局平移，世界坐标系下的位置
- `poses`: SMPL body pose，共 72 维 = 24 个关节 x 3 (axis-angle)。前 3 维是 root pose (全零，因为 root 旋转由 `Rh` 控制)
- `shapes`: SMPL shape 参数 (beta)，10 维 PCA 系数，所有帧共享

### 4.4 可视化图片

- **detec**: 各视角的 2D 检测框 + 关键点，拼接成一张图
- **repro**: 三角化 3D 关键点重投影回各视角，拼接成一张图
- **smpl**: SMPL 模型通过 pyrender 渲染叠加到各视角图片上 (需要 display 环境)
- **repro_smpl**: SMPL 模型关节点投影到各视角的可视化

所有拼接图使用 `merge()` 函数将多视角图像排列成网格。

---

## 5. 运行日志分析

以样例数据 `debug/test` 运行 3 帧为例:

```
  Demo code for multiple views and one person:
    - Input : debug/test => 1, 2, 3, 4        # 4 个相机
    - Output: debug/test/output_doc
    - Body  : smpl=>neutral, body25            # SMPL neutral, Body25 关键点

triangulation: 100%|██████████| 3/3             # 阶段一: 三角化
dump: 100%|██████████| 3/3                      # 写出 keypoints3d
Composing video: .../repro.mp4                  # ffmpeg 合成重投影视频

loading: 100%|██████████| 3/3                   # 阶段二: 加载数据
-> [Optimize global RT  ]:   1.4s               # Shape + RT 优化
-> [Optimize 3D Pose/3 frames]:  17.6s          # 3D Pose 优化 (L-BFGS)
-> [Optimize 2D Pose/3 frames]:   5.3s          # 2D Pose 优化 (L-BFGS)
render: ...                                     # 渲染 SMPL (需要 display)
```

---

## 6. 关键代码文件索引

| 文件 | 功能 |
|------|------|
| `apps/demo/mv1p.py` | 主入口: 三角化 + SMPL 拟合 |
| `easymocap/mytools/cmd_loader.py` | 命令行参数定义与解析 |
| `easymocap/dataset/mv1pmf.py` | `MV1PMF` 多视角单人数据集 |
| `easymocap/dataset/base.py` | `MVBase` 基类: 相机读取、数据加载、文件写入 |
| `easymocap/dataset/config.py` | Body25 等关键点配置 (关节数、骨架连接) |
| `easymocap/mytools/reconstruction.py` | 三角化 (`batch_triangulate`) 和重投影 (`projectN3`) |
| `easymocap/mytools/camera_utils.py` | 相机参数读写、畸变矫正 |
| `easymocap/mytools/file_utils.py` | JSON 读写、标注读取、结果写出 |
| `easymocap/mytools/writer.py` | `FileWriter`: 结果写出和可视化 |
| `easymocap/smplmodel/body_param.py` | SMPL 模型加载、参数工具函数 |
| `easymocap/smplmodel/body_model.py` | `SMPLlayer`: PyTorch SMPL 前向计算 |
| `easymocap/pipeline/basic.py` | `smpl_from_keypoints3d2d`: 多阶段优化编排 |
| `easymocap/pipeline/weight.py` | 各模型类型的默认损失权重 |
| `easymocap/pipeline/config.py` | `Config`: 优化控制参数 |
| `easymocap/pyfitting/optimize_simple.py` | `optimizeShape`, `optimizePose3D`, `optimizePose2D` |
| `easymocap/pyfitting/lossfactory.py` | 各种损失函数 (3D, 2D, 平滑, 正则) |
| `easymocap/visualize/renderer.py` | pyrender 渲染器 |

---

## 7. 数据流总结

```
输入:
  images/{cam}/*.jpg           各视角图片
  annots/{cam}/*.json          各视角 2D 关键点 (25 joints, Body25)
  intri.yml + extri.yml        相机参数

         │
         ▼
  ┌──────────────────────┐
  │  Phase 1: 三角化      │
  │  多视角 2D → 3D       │
  │  (SVD/DLT)           │
  └──────────┬───────────┘
             │
             ▼
  keypoints3d/{:06d}.json     3D 关键点 (25, 4)

         │
         ▼
  ┌──────────────────────────────────────────────┐
  │  Phase 2: SMPL 拟合                           │
  │  Step A: optimizeShape (骨长 → shapes)        │
  │  Step B: optimizePose3D (3D kpts → Rh,Th,poses)│
  │  Step C: optimizePose2D (2D kpts → 微调)      │
  └──────────┬───────────────────────────────────┘
             │
             ▼
  smpl/{:06d}.json            SMPL 参数: Rh(1,3), Th(1,3), poses(1,72), shapes(1,10)

输出可视化:
  detec/{:06d}.jpg            2D 检测可视化
  repro/{:06d}.jpg            三角化重投影可视化
  repro.mp4                   重投影视频
  smpl/{:06d}.jpg             SMPL 渲染可视化
  smpl.mp4                    SMPL 渲染视频
  repro_smpl/{:06d}.jpg       SMPL 关节重投影可视化
  repro_smpl.mp4              SMPL 关节重投影视频
```
