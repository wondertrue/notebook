# Notebook

个人学习笔记。当前仅维护本文件；后续会跨 workspace 补充各模块内容。

## Contents

- [深度学习基础知识](#深度学习基础知识)
  - [基础概念](#基础概念)
  - [网络结构](#网络结构)
  - [训练与优化](#训练与优化)
  - [笔记](#笔记)
- [计算机视觉基础知识](#计算机视觉基础知识)
  - [针孔相机与内参 K](#针孔相机与内参-k)
  - [ROS CameraInfo：K / R / P / D](#ros-camerainfok-r-p-d)
  - [投影矩阵 P](#投影矩阵-p)
  - [rectified 图像与畸变](#rectified-图像与畸变)
  - [笔记](#笔记-1)
- [特征点提取和匹配](#特征点提取和匹配)
  - [检测与描述](#检测与描述)
  - [匹配与几何验证](#匹配与几何验证)
  - [笔记](#笔记-2)
- [VLN](#vln)
  - [任务与设定](#任务与设定)
  - [方法](#方法)
  - [仿真器与可视化工具](#仿真器与可视化工具)
  - [Pipeline 文档](#pipeline-文档)
  - [笔记](#笔记-3)
- [3D-GS](#3d-gs)
  - [表示与渲染](#表示与渲染)
  - [训练](#训练)
  - [笔记](#笔记-4)
- [建图和定位](#建图和定位)
  - [GOReloc](#goreloc)
  - [FAST-LIO](#fast-lio)
  - [FAST-LIVO2](#fast-livo2)
  - [perception_oru NDT（D2D vs P2D）](#perception-oru-ndtd2d-vs-p2d)
  - [Cartographer](#cartographer)
- [maplab](#maplab)
  - [maplab console](#maplab-console)
  - [slam_node vertex 时间轴](#slam_node-vertex-时间轴)
  - [unitree project](#unitree-project)

---

## 深度学习基础知识

> 待补充：基础概念、网络结构、训练要点、常用资源。

### 基础概念

### 网络结构

### 训练与优化

### 笔记

---

## 计算机视觉基础知识

> 相机模型、坐标系、ROS 标定消息约定等通用视觉基础。

### 针孔相机与内参 K

- **针孔模型**：3D 点 \((X, Y, Z)\)（相机坐标系，Z 为光轴前方）投影到像素 \((u, v)\)：
  - \(u = f_x \cdot X/Z + c_x\)
  - \(v = f_y \cdot Y/Z + c_y\)
- **内参矩阵 K**（3×3，ROS `CameraInfo.K` 按行展开为 9 个数）：

```
K = [fx   0  cx]
    [ 0  fy  cy]
    [ 0   0   1]
```

- **maplab 标定 yaml** 中 `intrinsics.data: [fx, fy, cx, cy]`，与 K 一致。
- ROS 相机光学坐标系：**+X 右、+Y 下、+Z 前**（与 OpenCV / RealSense 一致）。

### ROS CameraInfo：K / R / P / D

`sensor_msgs/CameraInfo` 是 ROS 生态里最常用的相机标定消息，四个矩阵分工如下：

| 字段 | 尺寸 | 含义 |
|------|------|------|
| **K** | 3×3 | 内参（fx, fy, cx, cy） |
| **D** | 向量 | 畸变系数（模型由 `distortion_model` 指定，如 plumb_bob 5 个） |
| **R** | 3×3 | 校正旋转（rectification），把双目对齐到同一极平面 |
| **P** | 3×4 | **投影矩阵**，综合内参 + 外参，直接做 3D→像素投影 |

关系：**P = K · [R \| t]**（3×4），其中 \([R\|t]\) 把点从**校正后相机坐标系**变到像素齐次坐标。

### 投影矩阵 P

**定义**：3×4 矩阵，将校正后坐标系下的 3D 点 \((X, Y, Z)\) 投影为像素 \((u, v)\)：

\[
\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
=
\begin{bmatrix}
f_x' & 0 & c_x' & T_x \\
0 & f_y' & c_y' & T_y \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
\]

ROS 中 `P` 按**行优先**展开为 12 个数：`[P00, P01, P02, P03, P10, P11, P12, P13, P20, P21, P22, P23]`。

**单目 / 左目（参考相机）**：无立体偏移，\(T_x = T_y = 0\)，P 退化为 K 的形式：

```
P = [fx  0  cx  0]
    [ 0 fy  cy  0]
    [ 0  0   1  0]
```

**右目（立体对）**：P 的第 4 列 \(T_x\) 编码**基线 baseline**（两光心水平距离），ROS/OpenCV 约定：

\[
T_x = -f_x \cdot \text{baseline}
\qquad\Rightarrow\qquad
\text{baseline} = -T_x / f_x
\]

右目投影公式可理解为：\(u = f_x \cdot (X - B)/Z + c_x\)，其中 B 为基线。

**示例：Intel D435i 红外双目**（`camera_info.md`，640×480 rectified 流）：

| 相机 | fx | P 第 4 元素 \(T_x\) | 基线 |
|------|-----|---------------------|------|
| infra1（左） | 392.42 | 0 | — |
| infra2（右） | 392.42 | **-19.607** | \(-19.607 / 392.42 \approx 0.050\) m ≈ **50 mm** |

```yaml
# infra1 CameraInfo.P（行展开）
P: [392.42, 0, 323.01, 0,  0, 392.42, 246.97, 0,  0, 0, 1, 0]

# infra2 CameraInfo.P — 仅 Tx 不同
P: [392.42, 0, 323.01, -19.607,  0, 392.42, 246.97, 0,  0, 0, 1, 0]
```

两路 `frame_id` 均为 `camera_infra1_optical_frame`，`R = I`，说明图像已 rect 到**左目光学系**；此时 K 相同、D 全 0，右目相对位姿信息集中在 P 的 \(T_x\) 里。

**从 P 推 SLAM 外参**（maplab `T_B_C`）：已知左目 `T_B_C1` 和 baseline B，右目在左目右侧（+X 方向）：

```
T_C1_C2 平移 = [B, 0, 0]ᵀ   （R = I）
T_B_C2 = T_B_C1 · T_C1_C2
```

### rectified 图像与畸变

- topic 含 **`rect`**（如 `/camera/infra1/image_rect_raw`）表示驱动已做**去畸变 + 极线对齐**。
- 此时 `D = [0,0,0,0,0]`，可当作**零畸变 pinhole**；`distortion_model: plumb_bob` 只是字段名，系数为 0。
- 原始流（`image_raw`）通常有非零畸变，标定需区分 rect / raw。

### 笔记

> **来源** · 2026-09-03 · ROS CameraInfo P 矩阵与 D435i 双目 · chat: `dc42c8d2-bf63-4b72-9060-ebd0c054796b` · workspace: `slam-research`

---

## 特征点提取和匹配

> SuperPoint 检测描述 + LightGlue / SuperGlue 匹配；TensorRT / 高通部署差异。

### 检测与描述

### 匹配与几何验证

#### LightGlue TensorRT 部署（三实现对比）

**定位（别混）**

| 实现 | 是什么 |
|------|--------|
| **LightGlue**（`cvg/LightGlue`） | 论文原版 PyTorch 推理代码；自适应加速完整 |
| **LightGlue-ONNX**（`fabio-sim/LightGlue-ONNX`） | 把 LightGlue 改成可导出 ONNX，再跑 ORT / TensorRT / OpenVINO；**不是**「专门用 TensorRT 重写的 LightGlue」 |
| **本仓库导出器** `convert_lightglue_to_onnx.py` | 基于 LightGlue-ONNX 的 matcher-only，再改成现有 C++ SuperGlue ABI + TRT 8.5 |

数据流：图像 → **现有 SuperPoint** → 关键点+256维描述子 → LightGlue matcher。`superpoint_lightglue.pth` 只是「适配 SuperPoint 描述子的 matcher 权重」，不含 SuperPoint 网络。

产物：`dependencies/internal/superpoint-superglue/weights/fast_feature_matcher.onnx`（已核实 **9 层**，`transformers.0`…`8`，assignment 用 `log_assignment.8`）。C++：`light_glue.cpp` / `spsg_wrapper.cpp`。

**「层」不是 CNN feature map**

LightGlue 不卷积图像、不改变分辨率。每层是一组 **Transformer：self-attention + cross-attention**，在 **同一批关键点 token** 上反复更新 256 维描述子。800 个点进、800 个点出，没有「前面分辨率高、后面分辨率低」。

官方默认 9 层（不是 11；checkpoint 只有 9 层权重，设 11 会因 `strict=False` 多出随机层）：

```python
# LightGlue/lightglue/lightglue.py
"n_layers": 9,
```

本仓库导出：`--n_layers` 传给 `LightGlueMatcher`；已把 argparse 默认改成 9。`LightGlueMatcher.__init__` 里函数默认仍写 `n_layers: int = 6`，但 `main()` 会传入 `args.n_layers`，以命令行/argparse 为准。

```python
# convert_lightglue_to_onnx.py forward
for i in range(lg.n_layers):
    layer = lg.transformers[i]
    d0 = layer.self_attn(d0, enc0)
    d1 = layer.self_attn(d1, enc1)
    d0, d1 = _cross_block(layer.cross_attn, d0, d1)
scores = _match_assignment(lg.log_assignment[lg.n_layers - 1], d0, d1)
```

6 层可用（deep supervision 让中间层 assignment 也训练过），但困难图对匹配数会掉；最终 ONNX 用满 9 层。

**对比表**

| 点 | 通俗逻辑 | LightGlue | LightGlue-ONNX | 本仓库 `convert_lightglue_to_onnx.py` |
|----|----------|-----------|----------------|----------------------------------------|
| 输入 | matcher 吃点，不吃图 | 两图 keypoints + descriptors + image_size | 默认端到端：`images` → SuperPoint+matcher；也可 matcher-only | **仅 matcher**：`image0_size/image1_size` + kp/desc；SuperPoint 仍是独立 engine |
| 层数 | Transformer 堆叠次数，非特征图尺度 | 默认 9；可用 `--n_layers` 减层换速度 | 默认 9，固定跑完 | 默认现为 9；产物 ONNX 已是 9 层 |
| Adaptive depth | 简单图对提前停层 | 开，`depth_confidence=0.95` | **关**（forward 无 break） | **关**（`-1`） |
| Adaptive width | 后面几层丢掉「确定匹配不上」的点，降 attention 的 N² | 开，`width_confidence=0.99` | **关** | **关** |
| FlashAttention / compile | PyTorch CUDA 算子加速，数学不变 | `flash=True`，`torch.compile` | 导出用 SDPA；fused MHA 标 **ORT-only**，TRT 自己融子图 | `F.scaled_dot_product_attention`；TRT 8.5 能否融合看后端 |
| Assignment | SuperGlue 是多轮 Sinkhorn；LightGlue 是相似度+matchability+双向 softmax | 轻量头，无 Sinkhorn | 同左 | 同左，但改成 **双流** `d0/d1`，去掉 `0::2` slice |
| 两边点数 | query N0 vs base N1 | 可不等 | 堆成 `(2B,N,*)`，一般要求 N0=N1 | **dual-tensor**，允许 N0≠N1 |
| 输出 | K=有效匹配数 | dense `matches0[N0]` + 稀疏列表 | `matches[K,3]`、`mscores[K]`，**K 动态** | dense：`matches0[1,N0]` int32，无匹配填 **-1**；buffer 长度=N0，K 在 C++ 过滤 |
| 图像尺寸 | 只用于把像素坐标归一化到 [-1,1] 附近，shape 恒为 `[1,2]` | 两图可不同 | 可不同 | 支持不同；config 默认 query 536×960、db 648×864 |
| 部署目标 | | PyTorch CUDA | ORT / TRT / OpenVINO / FP8 | **TRT 8.5 C++**（opset 16，避免 LayerNorm op） |

**输出为何不是 `matches[K,3]`**

图内对每个 query 点写一个槽位：`valid` 则写匹配下标，否则 `-1`。C++ `light_glue.cpp` 按 `m=N0` 读满 buffer，再 `if (j < 0) continue` 得到长度为 K 的 `cv::DMatch`。高通若卡在「动态输出 K」，当前设计已避开；真正难的是 **输入 N0/N1 动态**（attention 中间张量 `[N0,N1]` 变）。

**原版加速 vs 部署能留下什么**

| 优化 | 原版 | ONNX/TRT/高通 |
|------|------|----------------|
| 提前停层 / 逐层砍点 | 有 | 控制流+动态 Gather，导出图里关掉 |
| 无 Sinkhorn 的 assignment | 有 | **保留** |
| 固定减层（9→6） | 可选 | 可选；本仓库最终仍用 9 |
| Flash / `torch.compile` | 有 | 不随 ONNX 带走；靠 TRT/QNN 融合与 FP16 |

**TensorRT 坑（已修）**

TRT 8.5 把 ONNX int64 输出静默写成 int32。C++ 若按 `int64_t*` 读，小端会把相邻两个 int32 拼成一个 int64，自匹配出现 **d≈2q**。现导出 `.to(torch.int32)`，C++ 用 `int32_t*`。

另一坑：LightGlue-ONNX 的 batch 维 swap + `[0::2]` 在 TRT 8.5 会算错；本仓库改 dual-tensor。

**高通 8550**

- 动态 **N0/N1**：HTP 常要静态 shape；可固定点数 + pad（必须 mask，否则假点污染 attention）。
- 动态 **K**：当前 dense 输出已不需要。
- 两图 **WxH 不同**：matcher 只吃两个 `[1,2]` 数值，不是动态 shape。若高通失败，更像常量折叠/只传一个 size/SuperPoint 图尺寸锁死。
- 不要为对齐尺寸把两图 **非等比** `cv::resize` 到 536×864。`cv::resize(src, dst, Size(W,H))` 是独立 `sx/sy`。更稳：letterbox（等比缩放 + **黑边 pad=0**）+ 丢掉 pad 区关键点 + 坐标/内参逆变换。

**相关路径**

- 官方：`LightGlue/lightglue/lightglue.py`
- ONNX 上游：`LightGlue-ONNX/lightglue_dynamo/models/lightglue.py`
- 导出：`dependencies/internal/superpoint-superglue/convert2onnx/convert_lightglue_to_onnx.py`
- 本地拷贝：同目录 `_lightglue_onnx.py`
- TRT C++：`.../src/light_glue.cpp`
- 计划：`.ai/plans/zh-CN/24-lightglue-onnx-export-boundary.md`、`25-lightglue-trt-int32-fix.md`

> **来源** · 2026-09-06 · LightGlue TensorRT 三实现对比 · chat: `c9ffa037-2426-4c4f-9773-3785ea0be8e4` · workspace: `rob_slam-research`

### 笔记

---

## VLN

Vision-Language Navigation。

> 待补充：任务定义、数据集、代表性方法、评测指标。

### 任务与设定

- **VLN-CE**（Vision-and-Language Navigation in Continuous Environments）：给机器人一句自然语言指令，只靠第一人称 RGB + 深度图 + 指令，在连续三维室内环境中一步一步导航到目标点。
- 数据集：R2R-VLNCE（Matterport3D 扫描房屋 + 人工标注指令）。
- 动作空间：前进 0.25m / 左转 15° / 右转 15° / 停止，共 4 个离散动作。
- 评测指标：SR（成功率）、SPL（路径效率加权成功率）、NE（终点到目标距离）、NDTW（轨迹贴合度）。

### 方法

- **CMA**（Cross-Modal Attention）：VLN-CE baseline 策略网络，由指令 LSTM 编码器 + 冻结 ResNet 视觉/深度编码器 + 跨模态注意力 + GRU 状态编码器 + 动作头构成。训练方式为 DAgger + Teacher Forcing 模仿学习。无 LLM、无 CoT、无 RL。
- **ABot-N1**：快慢双系统通用导航基座。慢系统为 VLM，做 CoT 推理，输出"像素目标"；快系统为轻量控制器，负责执行。五类导航任务统一为像素目标接口，产生跨任务正迁移。训练含大规模模仿预训练 + GRPO 强化学习后训练（下游导航奖励对齐）。

### 仿真器与可视化工具

- **Habitat-Sim**：VLN-CE 使用的三维仿真器（Facebook AI 开源），场景来自 Matterport3D 真实建筑扫描。
  - 场景文件：`.glb`（三维几何纹理）、`.navmesh`（可行走区域多边形网格）、`.house`（房间元数据）。
  - 当前本机 demo 场景：`data/scene_datasets/mp3d/17DRP5sb8fy/`
- **CARLA**：与 VLN-CE 无关，是自动驾驶领域的室外城市道路仿真器（Unreal Engine），支持摄像头/LiDAR/雷达，模拟车辆、行人、天气等。
- **habitat-viewer**（vlnce conda 环境自带）：加载 `.glb` + `.navmesh` 的三维交互查看器，按 `N` 显示可行走绿色网格。

```bash
conda activate vlnce
habitat-viewer \
  --navmesh-file data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh \
  data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb
```

- **Blender**（用于查看 `.glb`，与 vlnce 无关）：
  - 本机路径：`/home/maoli/utility/blender-3.3.21-linux-x64/blender`（3.3 版可兼容 Ubuntu 18.04 / GLIBC 2.27，5.2 版需要 GLIBC 2.28+ 不兼容）。
  - 打开 `.glb` 后若只看到外墙轮廓，在 Scripting 面板运行以下代码关闭背面剔除并看室内：

```python
import bpy
for mat in bpy.data.materials:
    mat.use_backface_culling = False
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.clip_start = 0.01
                space.clip_end = 10000
```

### Pipeline 文档

- VLN-CE + ABot-N1 完整 pipeline 文档（含流程图、模型架构、训练、代码指向、ABot-N1 替换对照表）已保存至：
  `notebook/assets/VLNCE and Abot-N1 pipeline.md`
- 文档共 11 章，章节大纲：
  1. 任务定义（VLN-CE 是什么）
  2. 整体 Pipeline 流程图（mermaid 图）
  3. 决策循环详解（每步 `policy.act → envs.step` 的代码走读）
  4. 智能体输入（RGB / Depth / instruction / 训练专用传感器）
  5. CMA 模型详解（指令 LSTM → 跨模态注意力 → GRU → 动作头）
  6. 自然语言指令的生命周期（指令不是实时生成，episode 内固定，附代码）
  7. 地图与导航（智能体无地图，navmesh 只在环境侧使用）
  8. 训练流程（DAgger + Teacher Forcing，R2R-VLNCE 数据集）
  9. 代码指向速查表
  10. 关键配置速查（分辨率、步长、词表、GRU 维度等）
  11. 现有模型清单 + ABot-N1 替换对照表（含术语速查：CoT、GRPO、像素目标、正迁移）

> **来源** · 2026-08-30 · VLN-CE 架构与 ABot-N1 技术对比 · chat: `f6ca8d63-d899-4184-ab78-c4b4121c2828` · workspace: `VLN-CE`

### 笔记

---

## 3D-GS

3D Gaussian Splatting。

> 待补充：表示与渲染、训练流程、变体与应用。

### 表示与渲染

### 训练

### 笔记

---

## 建图和定位

> 视觉 SLAM 中的建图（Mapping）与定位（Localization/Relocalization）方法，重点记录基于物体级别的系统。

### GOReloc

**GOReloc**（Graph-based Object-Level Relocalization）——RAL 2024 论文，基于 OA-SLAM / VOOM 扩展，用于视觉 SLAM 的**物体级重定位**。

- **核心思路**：用图结构描述物体之间的空间关系，通过图核（graph kernel）语义描述子，将当前帧的检测物体与地图中的 3D 物体做鲁棒关联，再用类 RANSAC 策略精化关联和位姿估计。
- **地图表示**：轻量的物体级地图，每个物体用**椭球体（Ellipsoid）**表示（位置、朝向、三轴尺寸）。
- **两阶段流程**：
  1. **建图**：运行 VOOM/OA-SLAM，生成含关键帧、ORB 地图点、物体椭球体的地图（`.yaml` + JSON）。
  2. **定位**：运行 `loc_tum`，加载地图，对新序列帧逐帧做重定位（`goreloc` 或 `points` 模式）。
- **关键代码位置**：
  - 图构建：`Tracking::GrabImageRGBD`（`src/Tracking.cc` ~L370）
  - GOReloc 核心：`Tracking::GOReloc`（`src/Tracking.cc` ~L2360）
  - 地图存取：扩展的 OSMAP 库（`src/Osmap.cc`）

**Viewer 中各显示层的含义**：

| 显示选项 | 内容 |
|----------|------|
| **Show Points** | 所有 ORB MapPoints。普通点用自身颜色；Reference MapPoints 红色；临时点（TmpPoints）绿色 |
| **Show Obj-Points** | 被归属于某个 Object 的 MapPoints，用该物体的颜色着色 |
| **Show Objects** | 每个 3D 物体的椭球体轮廓（线框），颜色随机或按类别着色；也可切换为 3D OBB（勾选 Disp 3D Bboxes） |

**ORB 与 YOLO 的分工**：

- **YOLO**（离线）：对图像做目标检测，输出 2D bbox + 类别 + 置信度，结果存为 JSON 文件。定位阶段直接读文件，不实时调用。
- **ORB**（在线）：提取图像关键点 + 描述子 → 三角化生成 3D MapPoints → 词袋重定位候选帧检索 → PnP 位姿解算。
- **融合方式**：在关键帧上，取落在 YOLO bbox 范围内的所有 ORB MapPoints（`kf->GetFeaturesInBox`），归属给对应 Object，用于拟合椭球体形状（`OptimizeReconstructionQuat`）。ORB 负责几何定位，YOLO 负责语义标注，二者互补。

> **来源** · 2026-08-30 · GOReloc 建图定位系统解析 · chat: `a1c26f79-dfe0-40cf-8152-ff92606389c1` · workspace: `GOReloc`

### FAST-LIO

**FAST-LIO**（Fast LiDAR-Inertial Odometry）——港大 Mars Lab 激光-惯性紧耦合里程计；本仓库以 **Livox MID360** + IMU 为主（`lidar_type: 1` → `avia_handler()`）。

**工作空间**：`/data/maoli/git/fast_livo_ws`（包路径 `src/FAST_LIO`）

**主文档**：

| 文档 | 路径 | 说明 |
|------|------|------|
| pipeline.md | `/data/maoli/git/fast_livo_ws/src/FAST_LIO/pipeline.md` | MID360 + bag `recording_20250918_083637`：驱动/消息、时间同步、IMU+ESKF、ikd-Tree 点到面、外参、RViz/PCD、排查与命令 |

**常用配置 / launch**（详见 pipeline §12）：

| 场景 | 配置 | launch |
|------|------|--------|
| MID360 + Livox 内置 IMU | `config/mid360.yaml` | `mapping_mid360.launch` |
| MID360 + RealSense `/camera/imu` | `config/mid360_cam.yaml` | `mapping_mid360_cam.launch` |
| LI-Init 标定 | `LiDAR_IMU_Init/config/mid360*.yaml` | `livox_mid360*.launch` |

**pipeline.md 章节概要**：

1. **一句话理解**：IMU 传播 → 去畸变 → 体素降采样 → ikd-Tree 近邻 → 点到面残差 → **迭代 ESKF** → 增量地图
2. **ROS 驱动与消息**：`livox_ros_driver2`、本仓最小消息包 vs 官方驱动、CustomMsg 字段、MID360 vs Velodyne、`scan_line: 4`
3. **数据接收与时间同步**：回调、`sync_packages`、`time_offset_lidar_to_imu` 符号约定
4. **IMU 初始化、积分与去畸变**：`IMU_init` / `UndistortPcl`、`curvature` 存点时间
5. **LiDAR 迭代 EKF 量测**：scan-to-map（非帧间 ICP）、`h_share_model`
6. **外参在线估计**：当前 `extrinsic_est_en: false`（用 LI-Init 固定外参）
7. **输出、RViz 与 PCD**：`/cloud_registered` vs ikd-Tree；`pcd_save` → `FAST_LIO/PCD/scans.pcd`（世界系 `camera_init`，`interval: -1` 单文件）
8. **RViz 圆环/散点**：MID360 扫描纹理、`currPoints` 累积、自反射 `blind`、时间/外参
9. **排查顺序**：暂停 bag、`surround` vs `currPoints`、`point_filter_num` / `blind`
10. **关键代码索引**：`preprocess.cpp`、`laserMapping.cpp`、`IMU_Processing.hpp`、`ikd-Tree`
11. **不包含**：回环、全局 PGO、重定位、PCD 闭环校正
12. **回放验证命令**：LI-Init / FAST-LIO（Livox IMU 与 camera IMU）

**PCD 在 CloudCompare 中上色**（`mid360_cam` 存盘同理，无相机 RGB）：

- 点类型 `PointXYZINormal`；有意义标量主要是 **`intensity`**（Livox `reflectivity`）
- Colors 选 **Scalar field** 且激活 `intensity` → 反射率；默认 Jet 色条一般为 **蓝=低、红=高**
- `normal_*`、`curvature` 在 FAST-LIO 存盘路径中多为 0，勿当几何法向使用

> **来源** · 2026-09-15 · FAST-LIO pipeline 文档索引与 PCD 着色 · chat: `a7612123-be1e-4918-af7d-88cffee70d53` · workspace: `fast_livo_ws`

### FAST-LIVO2

**FAST-LIVO2**（Fast, Direct LiDAR-Inertial-Visual Odometry）——港大 Mars Lab 的激光-惯性-视觉紧耦合里程计，视觉部分为**稀疏直接法**（sparse direct），不是稠密直接法。

**工作空间**：`/data/maoli/git/fast_livo_ws`

**环境依赖**（Ubuntu 18.04 + ROS Melodic）：

- OpenCV 4.5.4：`/home/maoli/opt/opencv-4.5.4`
- Sophus 非模板版 `a621ff`（需 `libSophus.so`，安装到 `/usr/local`）
- `rpg_vikit`（catkin 包，放 `src/`）
- 为消除 OpenCV 3.2/4.5 混用崩溃，工作空间内重编：
  - `vision_opencv/cv_bridge`（OpenCV 4）
  - `image_transport_plugins`（compressed / compressed_depth / theora）
- 不影响系统 `/opt/ros` 和其他 workspace（如 maplab），只在 `source fast_livo_ws/devel` 的终端生效

**编译命令**：

```bash
source /opt/ros/melodic/setup.bash
cd /data/maoli/git/fast_livo_ws
catkin_make -DOpenCV_DIR=/home/maoli/opt/opencv-4.5.4/lib/cmake/opencv4 -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

**运行命令**（以 HKU_Cultural_Center_01 为例）：

```bash
# 两个终端都先 source
source /opt/ros/melodic/setup.bash
source /data/maoli/git/fast_livo_ws/devel/setup.bash

# 终端 1：启动建图 + rviz
roslaunch fast_livo mapping_avia.launch

# 终端 2：播放 bag
rosbag play /data/maoli/git/fast_livo_ws/data/HKU_Cultural_Center_01.bag
```

**数据集与标定注意**：

- bag 话题：`/livox/lidar`、`/livox/imu`、`/left_camera/image`
- 每个序列的相机内参 + 相机-LiDAR 外参不同，需按 `data/calibration.yaml` 对应组修改：
  - `config/camera_pinhole.yaml`（内参）
  - `config/avia.yaml` 中 `Rcl` / `Pcl`（外参）
- `HKU_Cultural_Center_01` 用 calibration.yaml 中 **HKU 系列**那组，不是仓库默认的 CBD/Retail 组

**rviz / `/rgb_img` 中彩色点的含义**（`vio.cpp` → `plotTrackedPoints()`）：

- 底图：当前相机帧副本 `img_cp`
- 每个点：`visual_submap` 中的 `VisualPoint`，由 LiDAR 3D 点投影到当前图像
- 🟢 **绿色**（BGR `0,255,0`）：`errors ≤ propa_errors`，光度残差未变差 → **内点（inlier）**，参与 VIO 更新
- 🔵 **蓝色**（BGR `255,0,0`）：`errors > propa_errors`，优化后残差比 IMU 先验还差 → **外点（outlier）**，不可靠
- 蓝点成片通常意味着：外参/内参不准、遮挡、动态物体、光照变化

**视觉更新逻辑（patch + EKF）**：

1. **地图点来源**：3D 位置 `pt->pos_` 来自 **LiDAR 点云**，不由视觉三角化
2. **测量模型（patch 直接法）**：
   - 每个点存参考 patch（`ref_patch`），当前帧用仿射 warp 对齐
   - 残差 = 当前像素强度 − 参考 patch 强度（带逆曝光补偿）
   - 每个点贡献 `patch_size × patch_size`（默认 8×8=64）个光度残差，不是 2D 重投影
3. **估计器（迭代 ESIKF）**（`computeJacobianAndUpdateEKF` → `updateState`）：
   - 状态：`[3 旋转, 3 平移, 1 逆曝光]`（**不含地标 3D 坐标**）
   - 对每个金字塔层迭代：算 `H_sub`（光度残差对状态的雅可比）→ 卡尔曼增益更新位姿
   - 与 IMU 先验紧耦合：`vec = state_propagat - state`
4. **地图维护（非 EKF）**：
   - `generateVisualMapPoints`：从 LiDAR 点新建视觉点
   - `updateVisualMapPoints`：追加新观测（Feature = patch + 位姿）
   - `updateReferencePatch`：按观测质量更新参考 patch

**与 ROVIO（`ImgUpdate.hpp`）的对比**：

| 维度 | FAST-LIVO2 | ROVIO |
|------|-----------|-------|
| 观测模型 | 稀疏直接法光度 patch | 直接法 patch（`useDirectMethod_=true`） |
| 估计器 | 手写迭代 ESIKF | LWF 框架迭代 EKF |
| 地标是否在状态 | **否**，深度来自 LiDAR | **是**，估计 bearing + 逆深度 |
| 残差结构 | 全 patch 逐像素堆叠（`H_DIM = N×64`） | 每特征降维成 2D（`A_red_`, `b_red_`） |
| 坐标系 | world-centric | robocentric |
| 曝光估计 | 逆曝光进状态 | 无全局曝光状态 |
| 传感器 | LIVO（激光+IMU+相机） | 纯 VIO（IMU+相机） |

**patch 直接法 vs 光流特征+几何法**：

- 直接法优势：信息密度高（每点 64 个亚像素光度约束）、弱纹理可用、无描述子匹配、不确定度端到端建模
- 直接法劣势：亮度恒定假设脆弱、收敛盆地小（强依赖 IMU 先验）、对标定/外参/曝光敏感、外点剔除难
- 特征法优势：几何重投影误差更鲁棒、支持在线外参/td 自标定、RANSAC 剔除外点
- **标定精度低时优先特征法 + 在线外参估计**；直接法需先标定到位（Kalibr + FAST-Calib）

**稠密地图澄清**：

- 视觉 EKF 每帧只处理几十~几百个稀疏点（日志 `Retrieve N points`）
- rviz 中稠密彩色点云来自 **LiDAR 点云投影取色**（`dense_map_en`），不是视觉"造点"

> **来源** · 2026-08-31 · FAST-LIVO2 编译运行与视觉更新原理 · chat: `203305fd-8df6-444e-b266-1fa562b1aab1` · workspace: `fast_livo_ws`

### perception_oru NDT（D2D vs P2D）

**工作空间**：`/home/maoli/Data/unitree/IROS/ndt_ws`（`perception_oru` 包，`ndt_registration` / `ndt_fuser`）

**提出时间线**：

| 时间 | 工作 | 算法 |
|------|------|------|
| 2003 | Biber & Strasser, IROS | **P2D-NDT**（Point-to-Distribution），2D 激光 |
| 2007 | Magnusson 博士论文（Örebro） | **D2D-NDT**（Distribution-to-Distribution），推广到 3D |
| ~2009 | PCL 集成 | P2D-NDT（`pcl::NormalDistributionsTransform`） |
| 2010+ | `perception_oru` | 同时实现 P2D（`NDTMatcherP2D`）和 D2D（`NDTMatcherD2D` / `NDTMatcherD2D_2D`） |

**P2D-NDT（PCL 路线）**：

- 目标点云建 NDT 地图（体素含均值 μ、协方差 Σ）
- 源点云中每个点 pᵢ 查对应体素，score = exp(−0.5·(pᵢ−μ)ᵀ·Σ⁻¹·(pᵢ−μ))
- 对变换 T 求导，Newton 优化
- **单边**：只有目标侧是分布，源侧是原始点

**D2D-NDT（perception_oru / ndt_fuser 路线）**：

- 源、目标**各自**建 NDT 地图
- 匹配体素对 (Nₛ, Nₜ)：Δμ = T·μₛ − μₜ，合并协方差 Σ = Σₜ + R·Σₛ·Rᵀ
- score = exp(−0.5·Δμᵀ·Σ⁻¹·Δμ)
- **双边**：旋转也传播源侧 Σ，统计上更完整

**对比表**：

| 维度 | PCL P2D-NDT | perception_oru D2D-NDT |
|------|-------------|------------------------|
| 源侧表示 | 原始点 | NDT 体素（含 Σ） |
| 旋转对协方差 | 不处理源侧 Σ | R·Σₛ·Rᵀ 参与合并 |
| 多分辨率 | 支持 | 默认 2m→0.5m→0.2m 逐级精化 |
| 点云密度差异 | 较敏感 | 源侧 Σ 吸收稀疏噪声，更鲁棒 |
| 增量建图 | 目标 NDT 复用 | 天然分布对分布，适合 SLAM |
| 并行 | 部分 OpenMP | `ndt_matcher_d2d.cpp` 含 OpenMP |
| 计算量 | 较低 | 较高（两边都建图） |

**`ndt_fuser` 实际用法**：

- `matchLaser=true`（2D 激光）→ `NDTMatcherD2D_2D`
- 每帧激光先建局部 NDT（`ndlocal`），再与全局地图 D2D 配准
- `useOdometry=false` 时 Tmotion=Identity，主要靠 D2D 宽收敛域 + 多分辨率

**D2D 更优的典型场景**：

- 无可靠轮速/里程计，初始位姿差
- 2D 激光单帧稀疏（如 UTBM `/scan` ~540 rays）
- 地图与当前帧密度不一致的增量建图

**PCL 未集成 D2D 的原因（简述）**：

- 自动驾驶场景（高密度 Velodyne + IMU）下 P2D 已够用
- D2D 需重构数据结构（源侧也要 NDT Map），集成代价大
- `perception_oru` 团队未向 PCL 提 PR；PCL 长期低活跃
- Autoware 等对 P2D 做了大量工程优化（OpenMP/GPU），进一步降低迁移动机

> **来源** · 2026-08-31 · D2D-NDT 与 P2D-NDT 对比 · chat: `0b5efb9d-02fc-4dfa-888e-ee4e9b4269cf` · workspace: `perception_oru`

### Cartographer

**工作空间**：`/home/maoli/Data/unitree/IROS/cartographer_ws`

**文档索引**：

| 文档 | 路径 | 说明 |
|------|------|------|
| AGENTS.md | `/home/maoli/Data/unitree/IROS/cartographer_ws/src/AGENTS.md` | AI 助手仓库指南：工作空间布局、`catkin_make_isolated` 编译流程、模块架构与改代码清单 |
| paper_info.md | `/home/maoli/Data/unitree/IROS/cartographer_ws/src/paper_info.md` | Cartographer 原论文（Real-Time Loop Closure in 2D LIDAR SLAM）中英文摘录：闭环、SPA 优化、扫描匹配 |
| pipeline.md | `/home/maoli/Data/unitree/IROS/cartographer_ws/src/pipeline.md` | 2D Demo（`demo_backpack_2d.launch`）端到端数据流与源码走读 |
| tutorial.md | `/home/maoli/Data/unitree/IROS/cartographer_ws/src/tutorial.md` | 运行命令速查：2D/3D 建图、纯定位、landmarks demo |

**pipeline.md 章节概要**（`demo_backpack_2d.launch` + MultiEchoLaserScan + IMU）：

1. **启动架构**：Launch 层次（`demo_backpack_2d` → `backpack_2d` + rviz + rosbag play）、`backpack_2d.lua` 关键参数、ROS / 桥接 / Cartographer 核心三层架构
2. **端到端数据流**：rosbag → `cartographer_node` → SensorBridge → Collator → LocalTrajectoryBuilder2D → PoseGraph2D → TF 发布的时序图
3. **传感器接入**：订阅 `imu` / `echoes`；SensorBridge 做激光切分（10 段运动补偿）、TF 变换；Collator 按时间戳排序投递
4. **2D 点云位姿计算**（Local SLAM）：累积 10 帧 → 逐点 IMU 外推运动补偿 → 重力对齐 → Ceres scan matcher 对 active submap 优化 → `AddPose` 更新
5. **IMU 与雷达融合**：非经典 EKF；IMU 提供旋转外推与重力方向，雷达 scan match 通过 `AddPose()` 校正；平移外推来自历史匹配位姿差分
6. **后端 Pose Graph**：`AddNode` → 子图内约束 + 回环搜索 → SPA 全局优化（Ceres）
7. **输出与可视化**：TF `map→odom→base_link`、`/submap_list`、`/map`、`/scan_matched_points2` 等
8. **关键文件索引**：ROS 层 / Local SLAM / Global SLAM 源码路径速查
9. **总结**：2D 位姿 = 运动补偿 + 重力对齐 + Ceres 匹配；融合方式 = IMU 外推 + 扫描匹配校正

> **来源** · 2026-08-31 · Cartographer 文档索引与 pipeline 概要 · chat: `04c2d32e-c80f-4e0b-bc2e-4d5bc27c3b2a` · workspace: `src`

---

## maplab

> maplab / violoc 定位栈：坐标系约定、ROVIO 重力对齐、Unitree Go2 + D435i 工程笔记。

### maplab console

- [VIMap 数据结构、KFH、DL summary map](assets/maplab_console_note.md)
- 独立笔记涵盖 `Vertex`、`VisualNFrame` 像素 vs Landmark、`Edge`、
  `KeypointIdentifier`、`LandmarkStore`、`LandmarkIndex`、KFH 删除链路，
  以及 **DL localization summary map**（observer、像素、无 landmark 反向表）。
- 暂不整理回环检测、landmark 合并、以及 summary 的 Binary 通道。

### slam_node vertex 时间轴

VI-Map **vertex 时间基准是图像 NFrame，不是 odometry**。`slam_odom_T_M_I` 只提供轨迹采样，在 `t_image` 上插值得到 vertex pose；每条 odometry **不**创建 vertex。

```cpp
// applications/maplab-node/src/mapping/map-update-builder.cc
const int64_t timestamp_nframe_ns =
    tracked_nframe->nframe->getMinTimestampNanoseconds();
T_M_B_buffer_.interpolateViNodeStateAt(
    timestamp_nframe_ns, &(map_update->vinode));
```

三条时间轴各自保留原始 stamp，靠插值关联：

| 轴 | 作用 |
|----|------|
| `t_img` | 创建 vertex；SPSG / image descriptor 按最近 vertex 挂载（tolerance 默认 0.5 ms） |
| `t_odom` | 写入 `PoseLookupBuffer`，给任意 `t_img` / `t_lidar` 插值 |
| `t_lidar` | sensor resource（scan-end）；不必等于 `t_img`/`t_odom`；导出时在 `t_lidar` 插值 `T_G_L` |

odometry 比图像多（例如 ImgUpdate + LidarUpdate 都发 odom → ~20 Hz）是正常的：多出来的样本只加密插值，不产生空 vertex，也不要求对应 descriptor。

> **来源** · 2026-09-17 · VI-Map vertex 时间轴 · chat: `0c3a2617-73a2-4612-8a0d-dde0addcf7c0` · workspace: `rob_slam-research`

### unitree project

**工程上下文**：`maplab_ws` + `dog-violoc.launch`，标定 `d435i.yaml`，IMU topic `/camera/imu`，相机 `/camera/infra1/image_rect_raw`。

**IMU / Body 坐标系（RDF，与相机光学系一致）**：

- `+X`：右，`+Y`：下，`+Z`：前（右-下-前）。
- `d435i.yaml` 中 body 基准 = IMU（`T_B_S` 单位阵），相机 `T_B_C ≈ I`，故 ROVIO 的 body/IMU 帧 ≈ D435i 光学系。
- bag 静止段加速度计读数约 `(0.1, -9.6, 0)` m/s²，主分量在 **-Y**，与「Y 朝下、比力向上」一致；`rovio-flow.cc` 原样转发 IMU，不做轴交换。

**重力对齐（ROVIO 如何保证世界系 Z-up）**——三处代码，缺一不可：

1. **假设**：`ImuPrediction.hpp` 硬编码 `g_ = (0, 0, -9.81)`，世界 **+Z = 反重力（上）**。
2. **落地**：`FilterStates::initWithAccelerometer()` 用 `qWM.setFromVectors(fMeasInit, unitZ)`，把静止加速度计测到的「上」对齐到世界 +Z（只定 roll/pitch，**yaw 不定**）。
3. **触发**：`RovioInterfaceImplInl.hpp` 首帧 IMU、`WaitForInitUsingAccel` 时调用 `resetWithAccelerometer()`。

**I 系（全局地图系）到底规定了什么**：

- ✅ 明确：**+Z 朝上**（重力对齐）。
- ❌ 未明确：水平面 +X/+Y 是「前左（FLU）」还是「右前（RDF）」；建图起点 yaw 由初始化/当时车体朝向决定，代码无「+X=车头」硬编码。
- ROVIO 帧注释：`W` = VIO 局部系，`I` = 全局地图系，`M` = IMU 机体系；定位通过 `poseLin(0)`/`poseRot(0)`（`IrIW`/`qWI`）把 W 对齐到 I。

**`$LocPose` / `att` 里 roll≈-90° 的粗略解释**：

- `$LocPose`（`Update.hpp`）打印的是 **IMU(M) 在全局地图系(I) 中的位姿**：`tIM` = 位置，`att` = `qIM` 的 RPY。
- `DCM2Euler_RPY` 注释写 FLU（x前 y左 z上），但这是**解欧拉角的数学约定**；被描述的对象仍是 **RDF 的 IMU**。
- 相机水平朝前静止时：IMU +Y（下）对应世界 -Z（下）→ 绕 X 约 **-90° roll**；pitch≈0、yaw 才有物理航向意义。故 roll≈-90° **不是 bug**，是光学系/IMU 系相对 Z-up 世界系的固定偏置。
- **位置** `xyz(tIM)`：地图 I 系下的水平 (x,y) + 高度 z，语义正常；不是车体 FLU 下的「前/左/上」分量。
- 若产品输出要「静止平地 roll/pitch≈0 的 FLU 车体角」：勿改 ROVIO 内部 `g_`；在输出层用 `d435i.yaml` 的 `target_point_extrinsics` / `T_B_P`（RDF→FLU 固定旋转）得到 `T_G_P` 再解姿态。`$FusePose` 已用 `T_G_P`；`MM_Inject` 若仍从 `T_G_B` 解角会依旧显示 -90°。

**相关文件**：

| 主题 | 路径 |
|------|------|
| 重力常量 | `dependencies/internal/maplab_rovio/include/rovio/ImuPrediction.hpp` |
| 加速度计初始化 | `dependencies/internal/maplab_rovio/include/rovio/FilterStates.hpp` |
| RPY 工具 | `dependencies/internal/maplab_rovio/include/rovio/attitude.hpp` |
| `$LocPose` 打印 | `dependencies/internal/maplab_rovio/lightweight_filtering/include/lightweight_filtering/Update.hpp` |
| Unitree 标定 | `applications/maplab-node/launch/euroc/calib/d435i.yaml` |
| RDF→FLU 输出外参 | `target_point_extrinsics` / `common/vio-common/target-point-config.h` |

> **来源** · 2026-09-02 · maplab violoc IMU 系与重力对齐 · chat: `c272ab80-b4aa-4cfc-9cac-2a5e6b771b4b` · workspace: `slam-research`
