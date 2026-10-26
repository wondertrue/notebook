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
  - [PoseLookupBuffer 与 IMU 前向积分](#poselookupbuffer-与-imu-前向积分)
  - [ROVIO 三组隐式位姿状态](#rovio-三组隐式位姿状态)
  - [建图同步与 LiDAR 时间对齐](#建图同步与-lidar-时间对齐)
  - [ROVIO LiDAR Jacobian、信息矩阵与虚拟量测](#rovio-lidar-jacobian信息矩阵与虚拟量测)
  - [Map Tracking 虚拟 database 帧](#map-tracking-虚拟-database-帧)
  - [unitree project](#unitree-project)
- [Docker](#docker)
  - [用 Docker 修复宿主机目录所有权](#用-docker-修复宿主机目录所有权)
  - [docker run](#docker-run)
  - [docker exec](#docker-exec)
- [C++语法规则](#c语法规则)
  - [原子比较交换 compare_exchange_strong](#原子比较交换-compare_exchange_strong)

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

#### KFH 默认参数与 dense recipe

`kfh` 沿 mission 顺序遍历 vertex；满足任一条件就把当前 vertex 选为
keyframe：达到最大帧间隔、与上一 keyframe 的共视 landmark 太少、有绝对
6DoF 量测、有 loop-closure edge，或相对位移/旋转达到阈值。首尾 vertex
会保留；其余非 keyframe vertex 的视觉信息被删除，IMU edge 则拼接。

参数方向：

- `kf_distance_threshold_m`：相对上一 keyframe 的位移达到阈值就保留。
  **越小越容易触发，保留越密**。
- `kf_rotation_threshold_deg`：相对旋转达到阈值就保留。
  **越小越容易触发，保留越密**。
- `kf_every_nth_vertex`：未触发其他条件时允许连续跳过的数量上限。
  **越小越早强制保留，保留越密**。
- `kf_min_shared_landmarks_obs`：计算**当前 vertex 与上一个已选 keyframe**
  共同观测到的 landmark 数；不是与时间上的上一帧比较。判定条件为
  `common_observations < threshold`，成立时保留当前 vertex：
  - 默认值 20：共视数 `0–19` 时保留，`>=20` 时不由此条件保留。
  - dense 值 15：共视数 `0–14` 时保留，`>=15` 时不由此条件保留。
  - 共视数为 0 时仍会保留。设计意图是把“与上一 keyframe 重叠很少”的
    新视角或新区域作为 keyframe，维持地图覆盖与 track 连续性；该指标不是
    图像质量或弱帧过滤器。
  - 如果共视为 0 的原因是跟踪失败、遮挡或当前帧本身没有有效 landmark，
    现有启发式同样会保留这个弱帧。要过滤这种情况，需要另加“当前帧有效
    landmark 总数/质量”的门槛，不能只调本参数。
  - 因为阈值越大，`common < threshold` 越容易成立，所以**越大越容易保留、
    keyframe 越密**。这一方向与参数名及直觉都容易混淆。

裸跑 `kfh` 的代码默认值：

```text
distance=0.75 m, rotation=20 deg, every_nth=10, min_shared=20
```

`cfg/base/mapping/console/kfh_dense.commands`：

```text
distance=0.2 m, rotation=10 deg, every_nth=5, min_shared=15
```

对比结论：

- dense 的距离、旋转、帧间隔三个阈值都更容易保留 keyframe，因此整体上
  **比默认更密、剪枝更弱**。
- dense 的 `min_shared` 从 20 降到 15，反而更难因“共视不足”保留 keyframe，
  这一项单独看比默认更稀；它与另外三个参数方向相反。
- 所以不能笼统说 dense 的每个条件都“更严格”。若“严格”指删除更多 vertex，
  **默认 `kfh` 整体更严格/更激进**；若指 keyframe 保留更敏感，则 dense
  整体更敏感。
- Go2 实测默认 `kfh` 保留 `471/4039 = 11.7%`，删除 88.3%。轨迹平稳且
  相邻帧共视高时，距离、旋转、共视条件较少触发，结果主要受
  `kf_every_nth_vertex=10` 控制。
- `kfh` 会原地修改内存地图；已经删除的 vertex 不能靠再跑 dense 参数恢复。
  对比两组参数必须从同一份原始 VI-Map 分别重新 `load` 后运行。

> **来源** · 2026-10-14 · KFH 默认与 dense 参数对比 · chat: `fef9e509-9ec4-4d77-bfea-b85b11dd7a46` · workspace: `rob_slam-research`

#### ELQ 标记与 summary landmark 筛选

`elq` 的调用链：

1. `console-plugins/landmark-manipulation-plugin/src/landmark-manipulation-plugin.cc`
   中，console 命令 `elq` 调用
   `vi_map_helpers::evaluateLandmarkQuality(mission_ids, map)`。
2. `algorithms/vi-map-helpers/src/vi-map-landmark-quality-evaluation.cc`
   遍历 mission 内全部 landmark，并以 `kReEvaluateLandmarkQuality=true`
   强制重算质量：

```cpp
landmark.setQuality(
    vi_map::isLandmarkWellConstrained(
        *map, landmark, kReEvaluateLandmarkQuality)
        ? vi_map::Landmark::Quality::kGood
        : vi_map::Landmark::Quality::kBad);
```

3. `map-structure/vi-map/src/landmark-quality-metrics.cc` 中，
   `isLandmarkWellConstrained()` 按特征类型分流：LiDAR landmark 走
   `isLiDARLandmarkWellConstrained()`，其余视觉 landmark（包括 Binary
   和 DL）走 `isVisualLandmarkWellConstrained()`。视觉检查包括最少观测数、
   最近观测距离、最大视差角，以及启用时的重投影误差。

summary 导出的筛选链：

1. `console-plugins/vi-map-summarization-plugin/src/summarization-plugin.cc`
   的 `saveSummaryMapToDisk()` 调用
   `createLocalizationSummaryMapForWellConstrainedLandmarks()`。
2. `map-structure/localization-summary-map/src/localization-summary-map-creation.cc`
   先调用 `VIMapQueries::getAllWellConstrainedLandmarkIds()`，再把筛选后的
   landmark ID 列表传给 `createLocalizationSummaryMapFromLandmarkList()`。
3. `algorithms/vi-map-helpers/src/vi-map-queries.cc` 遍历全部 landmark，只保留
   `isLandmarkWellConstrained()` 返回 true 的 ID。已有 `Good/Bad` 标记会直接
   判定；只有 `Unknown` 才现场计算几何质量。
4. Binary 与 DL 类型的拆分发生在上述统一筛选之后，因此 `elq` 标成 `Bad`
   的 DL landmark 同样不会写入 summary，也不会进入后续 DL 匹配与 PnP。

结论：`elq` 不是只为 `kfh` 服务；它在 `rtl` 后统一刷新 landmark 的几何质量，
而 summary 只导出通过该质量门槛的 landmark。

> **来源** · 2026-10-14 · ELQ 与 summary 筛选链路 · chat: `2ac66d53-0a49-4d5a-84a8-af965fcfeae1` · workspace: `rob_slam-research`

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

### PoseLookupBuffer 与 IMU 前向积分

`vio_common::PoseLookupBuffer`（`common/vio-common/src/pose-lookup-buffer.cc`）在 buffer 内 odometry 时间序列上为任意时刻取 `ViNodeState` / `T_M_I`。**`interpolateViNodeStateAt`** 是建图主入口（`MapUpdateBuilder`）；定位 **`rob_slam_loc`** 用的是同类的 **`getPoseAt`**（`RovioLocalizationHandler::T_M_I_buffer_`），逻辑与下面「向前 IMU 分支」对称，但 violoc **不调用** `interpolateViNodeStateAt`。

**`interpolateViNodeStateAt` 决策顺序（易记：不是「超 500 ms 才 RK4」）**

1. buffer 空 / 请求时刻早于最老样本 → 失败（`kFailedNotYetAvailable` / `kFailedWillNeverSucceed`）。
2. **`buffer_odometry_estimate_.interpolateAt` 成功** → 相邻两帧 odometry 之间 **线性插值位姿 + slerp 四元数**（`interpolateTransformation`），**不用 RK4**；返回 `kSuccessInterpolated`。
3. 插值失败且 **`timestamp_ns` 晚于 buffer 最新 odometry**（需向前补）：
   - 令 `Δt = timestamp_ns - t_newest`。
   - **若 `Δt > max_allowed_propagation_time_ns_`** → **`kFailedNotYetAvailable`，不做 IMU 积分**。
   - **若 `Δt ≤ max_allowed_propagation_time_ns_`** → 调用 **`vio::propagatePoseUsingImu`**（`imu-forward-propagation.cc`），内部 **`imu_integrator::ImuIntegratorRK4::integrateStateOnly`** 逐步积分；返回 `kSuccessImuForwardPropagation`。积分后四元数 **未** 在 buffer 层做 `normalize()`。

**500 ms 上限从哪里来**

| 用途 | 构造处 | 默认值 |
|------|--------|--------|
| **建图** `Synchronizer::T_M_B_buffer_` | `synchronizer.cc` 传入 gflag | **`--odometry_buffer_max_forward_propagation_ns`**，默认 **`aslam::time::milliseconds(500)`（0.5 s）** |
| **定位** `RovioLocalizationHandler::T_M_I_buffer_` | `rovio-localization-handler.h` | 编译期 **`kBufferMaxPropagationNs = aslam::time::milliseconds(500)`**，无 gflag |

建图 gflag 定义：

```cpp
// applications/maplab-node/src/mapping/synchronizer.cc
DEFINE_int64(
    odometry_buffer_max_forward_propagation_ns, aslam::time::milliseconds(500),
    "Determines the maximum duration the odometry buffer can "
    "forward-propagate using the IMU.");
```

**相关调用**

- 建图：`T_M_B_buffer_.interpolateViNodeStateAt(t_nframe, &map_update->vinode)`（见上节 `map-update-builder.cc`）。
- 定位：`T_M_I_buffer_.getPoseAt(loc_ts, &T_M_I)` → 重力门控 / `T_G_I` / MM 注入等（**不再**二次 lookup）。
- **`getRotationBetween`**（特征跟踪帧间旋转）：只用 IMU 陀螺积分，**不走** `max_allowed_propagation_time_ns_` 上限。

> **来源** · 2026-09-22 · PoseLookupBuffer 500 ms 与 RK4 分支 · chat: `eb882abe-dff4-4ed5-93dc-38f3d903199d` · workspace: `rob_slam-research`

### ROVIO 三组隐式位姿状态

ROVIO 当前有三组容易混淆的附加位姿状态。前两组由 `nPose` 控制并复用
`_pop/_poa` 数组，第三组由 `enableMapLocalization` 独立控制。

#### 1. 惯性系对齐状态 `IrIW/qWI`

- `IrIW`：ROVIO 局部世界系 W 原点在外部全局惯性系 I 中的位置。
- `qWI`：W 与 I 之间的旋转。
- 存储在 `state.poseLin(inertialPoseIndex_)` 与
  `state.poseRot(inertialPoseIndex_)`，对应 `_pop/_poa` 的一个元素。
- `PoseUpdate` 6DoF 路径使用外部 `T_G_B` 量测更新这组状态。
- maplab 命名中 ROVIO 的 W 对应 mission 系 M，I 对应 global map 系 G；
  输出按下式直接构造：

```text
T_G_M = Transformation(qWI.inverse(), IrIW)
```

#### 2. 量测 body 外参状态 `MrMV/qVM`

- ROVIO `PoseUpdate` 的 V 是外部位姿量测所描述的 body/sensor frame，M 是
  ROVIO IMU frame。
- `MrMV/qVM` 表示 IMU M 与量测 body V 之间的外参。
- `bodyPoseIndex_ >= 0` 时，它们存储在 `_pop/_poa` 的另一个元素中，进入
  covariance，并由 `PoseUpdate` 联合估计。
- `bodyPoseIndex_ == -1` 时，不存在对应状态维；`MrMV/qVM` 只是
  `PoseUpdate` 内的固定配置参数。相关配置仍存在，但
  `init_cov_MrMV/init_cov_qVM` 和 `pre_cov_MrMV/pre_cov_qVM` 不会注册到
  filter state。

`nPose` 与两个索引的当前模板关系：

```text
nPose = 0:
  inertialPoseIndex_ = -1
  bodyPoseIndex_     = -1
  _pop/_poa 中没有 additional pose

nPose = 1:
  inertialPoseIndex_ = 0
  bodyPoseIndex_     = -1
  状态仅含 IrIW/qWI，共 6 维误差状态

nPose = 2:
  inertialPoseIndex_ = 0
  bodyPoseIndex_     = 1
  状态含 IrIW/qWI 与 MrMV/qVM，共 12 维误差状态
```

这里真正决定状态维数的是 `nPose`；两个 index 只是说明每个数组槽位的用途。

#### 3. 结构特征定位状态 `WrWG/qWG`

- 由 `enableMapLocalization` 控制，对应独立的 `_pmp/_pma`，不占用
  `nPose` 的 `_pop/_poa`。
- `LocalizationLandmarkUpdate` 使用图像 2D keypoint 与地图 3D landmark
  约束更新该状态。
- 它保存的是 `T_M_G` 方向，输出时求逆：

```text
T_M_G = Transformation(qWG, WrWG)
T_G_M = T_M_G.inverse()
```

#### 三组状态的关系

`IrIW/qWI` 与 `WrWG/qWG` 在物理上都描述 global map 与 local mission 的
对齐关系，但方向、状态槽位和测量模型不同：

```text
IrIW/qWI → PoseUpdate 6DoF 路径
          → 直接导出 T_G_M

WrWG/qWG → LocalizationLandmarkUpdate 结构特征路径
          → 保存 T_M_G，再求逆得到 T_G_M
```

运行时由 `violoc_use_6dof_localization` 选择其中一条定位路径。当前工厂同时
设置 `kNPoses=1` 和 `kEnableMapLocalization=true`，因此两类状态在编译后的
filter state 中都存在；实际导出哪一组取决于运行模式。`MrMV/qVM` 当前因为
`bodyPoseIndex_=-1` 仅作为固定外参使用。

> **来源** · 2026-10-15 · ROVIO 三组位姿状态 · chat: `f5164701-4633-40e2-9b1d-ea2fd2ed9ba5` · workspace: `rob_slam-research`

### 建图同步与 LiDAR 时间对齐

#### Vertex 与图像派生测量

- VI-Map vertex 由经过跟踪的 `VisualNFrame` 创建；不是因为 SPSG/VPR 输入图像才创建。SPSG external features 和 VPR image descriptor 是附加到既有 vertex 的图像派生测量。

```cpp
// map-update-builder.cc: VisualNFrame 决定 MapUpdate 的时间戳，
// odom 状态被插值到该图像时刻。
const int64_t timestamp_nframe_ns =
    tracked_nframe->nframe->getMinTimestampNanoseconds();
T_M_B_buffer_.interpolateViNodeStateAt(
    timestamp_nframe_ns, &(map_update->vinode));
map_update->keyframe = tracked_nframe;

// stream-map-builder.cc: MapUpdate 中的 NFrame 被放进新 vertex。
vi_map::Vertex* map_vertex = new vi_map::Vertex(
    vertex_id, vinode_state.getImuBias(), nframe, invalid_landmark_ids,
    mission_id_);
map_vertex->set_T_M_I(vinode_state.get_T_M_I());
map_->addVertex(vi_map::Vertex::UniquePtr(map_vertex));
```

- 两类附加测量必须保留源图像的 `header.stamp`，再按最近 vertex 挂载。当前容差取 `vio_nframe_sync_tolerance_ns = 500000 ns`（0.5 ms），其用途是容纳同一帧的微小时间误差，不允许跨相机帧匹配。

#### Mapping Synchronizer 的驱动关系

- `Synchronizer::processOdometryMeasurement()` 每收到一条 odom，先写入 `T_M_B_buffer_`，随后调用 `releaseData()`；因此 **odom 是数据释放的驱动源**，但 vertex 仍由图像 NFrame 创建。

```cpp
// synchronizer.cc
DEFINE_int64(
    odometry_buffer_history_ns, aslam::time::seconds(30u),
    "History length of the buffered external 6DOF odometry measurements.");

void Synchronizer::processOdometryMeasurement(
    const vio::ViNodeState& odometry) {
  T_M_B_buffer_.bufferOdometryEstimate(odometry);
  releaseData();
}
```

- `odometry_buffer_history_ns = 30 s` 是**已收到 odom 的历史保留长度**。领先当前最新 odom 的 LiDAR 会留在 LiDAR buffer 等待，Synchronizer 没有单独的 wall-clock 等待超时；持续收到更新 LiDAR 时，旧数据会因 `30 s` 时间跨度或超过 `256 scans` 被淘汰。更上游的 Go2 `lidar_resource_node` 另有 `max_wait_s: 2.0`，用于等待一帧 scan 所需的 IMU/odom 覆盖。
- 可释放 pose 窗口是 IMU 与 odom 可用区间的交集：
  - `oldest = max(oldest_odom, oldest_imu)`
  - `newest = min(newest_odom, newest_imu)`
- camera、LiDAR 等测量小于 `oldest` 时因无法恢复 pose 而丢弃；时间戳不大于 `newest` 时释放；大于 `newest` 时继续等待后续 IMU/odom。
- LiDAR 不要求靠近图像或 vertex。`synchronizer_lidar_buffer_length_ns = 30 s` 和 `synchronizer_lidar_buffer_max_scans = 256` 只是缓存保护，不是 LiDAR-image 同步容差。

#### LiDAR 去畸变、存图与 console 放置

1. `lidar_resource_node` 使用覆盖整帧 `[scan_begin, scan_end]` 的 IMU，并把 odom 插值到 `scan_end`；随后将每个点从自己的采样时刻补偿到 `L(scan_end)`。输出点云时间戳为 `scan_end`。
2. `slam_node` 将该点云按 LiDAR sensor ID 和 `scan_end` 保存为 **mission 级 temporal sensor resource**。它不附着在任何 vertex 上，也不是 vertex 的 frame resource：

```cpp
// stream-map-builder.h
map_->addSensorResource(
    point_cloud_type, lidar_sensor_id, timestamp_ns, point_cloud, &mission);

// vi-map-inl.h
mission->addSensorResourceId(
    type, sensor_id, resource_id, timestamp_ns);
```

3. console 导出或显示时，`PoseInterpolator` 从相邻 vertex 状态出发，沿 VIWLS IMU edge 使用 `ImuIntegratorRK4` 积分到点云自己的 `t_lidar`，得到连续位姿 `T_M_B(t_lidar)`。
4. 点云最终按下式放入全局地图：

\[
p_G = T_{G\_M}\,T_{M\_B}(t_{lidar})\,T_{B\_L}\,p_{L(scan\_end)}
\]

- 这里是“在 LiDAR 时间戳计算 pose”，不是“把 LiDAR 转到对应 vertex 时间戳”。只要求 `t_lidar` 落在 VIWLS 轨迹可插值范围内；范围外的资源会跳过。
- 后端优化后的 vertex 状态会改变 scan 的全局刚体放置；但帧内去畸变仍使用建图在线阶段的 IMU/odom，不会在 console 中按优化轨迹重新逐点去畸变。

> **来源** · 2026-09-18 · 建图同步与 LiDAR 时间对齐 · chat: `403b0ae0-d528-4872-992e-14a0dbf8e9c5` · workspace: `rob_slam-research`

### ROVIO LiDAR Jacobian、信息矩阵与虚拟量测

#### Jacobian、Hessian 与信息矩阵

设原始残差为 \(r\)，对 6DoF pose 的 Jacobian 为 \(H\)，量测噪声 covariance
为 \(R\)。用 \(R^{-1/2}\) 白化：

\[
l=R^{-1/2}r,\qquad J=R^{-1/2}H.
\]

白化后噪声 covariance 为 \(I\)。当前代码累计：

\[
HtH=J^\mathsf TJ=H^\mathsf TR^{-1}H,
\]
\[
Htl=J^\mathsf Tl=H^\mathsf TR^{-1}r,\qquad
ltl=l^\mathsf Tl.
\]

- \(J\) 是 Jacobian，约为“点数 × 6”。
- \(HtH\) 是 Gauss-Newton 近似 Hessian，也是该批量测对 pose 提供的
  **information matrix**。
- 信息矩阵不是 Jacobian；一般量测信息为 \(H^\mathsf TR^{-1}H\)，状态 prior
  信息矩阵为 \(P^{-1}\)。

#### 为什么同时需要 \(P_{pp}\) 和 \(P_{xp}\)

- \(P_{pp}\)：pose 相对 pose 的 6×6 covariance，用于创新 covariance：
  \[
  P_y=I+B P_{pp}B^\mathsf T.
  \]
- \(P_{xp}\)：完整状态相对 pose 的交叉 covariance，尺寸为“完整状态维数 × 6”，
  用于 Kalman gain：
  \[
  K=P_{xp}B^\mathsf TP_y^{-1}.
  \]

LiDAR 只直接观测 pose；\(P_{xp}\) 将这份信息传给 velocity、gyro/accel bias
等相关状态。若只更新 pose，\(P_{xp}\) 对应的 pose 行就是 \(P_{pp}\)；若使用
完整状态 Jacobian，则 \(P_{xp}\) 隐含在 \(P H_{\text{full}}^\mathsf T\) 中。

#### 为什么不能直接把 \(HtH/Htl\) 当作 \(H/r\)

若白化后的原始模型为：

\[
l=J\delta p+e,\qquad e\sim\mathcal N(0,I),
\]

左乘 \(J^\mathsf T\) 后：

\[
Htl=HtH\,\delta p+\eta,\qquad
\operatorname{Cov}(\eta)=HtH.
\]

因此把 `Htl` 当作 6 维虚拟残差时，正确形式是
\(H_v=HtH,\ R_v=HtH\)，不是 \(R_v=I\)：

- 错用 \(H_v=HtH,\ R_v=I\) 会把信息再次平方为 \(HtH^2\)，信息向量变成
  \(HtH\cdot Htl\)。
- 使用 \(H_v=HtH,\ R_v=HtH\) 虽然理论等价，但退化环境中 `HtH` 会降秩；
  \(HtH\,P_{pp}\,HtH+HtH\) 与 `HtH` 具有相同零空间，无法做正定 LLT。

#### 特征分解构造标准虚拟量测

当前 `ConstructVirtualMeasurement()` 对称特征分解：

\[
HtH=V\Lambda V^\mathsf T.
\]

对每个机器精度以上的正特征值 \((\lambda_i,v_i)\)，构造：

\[
B_i=\sqrt{\lambda_i}\,v_i^\mathsf T,\qquad
y_i=\frac{v_i^\mathsf T Htl}{\sqrt{\lambda_i}}.
\]

于是：

\[
B^\mathsf TB=HtH,\qquad B^\mathsf Ty=Htl,
\]

并可使用单位噪声的标准 measurement-form Kalman 更新：

\[
P_y=I+B P_{pp}B^\mathsf T,
\]
\[
K=P_{xp}B^\mathsf TP_y^{-1},
\]
\[
\delta x=d-K(y+Bd),\qquad
P^+=P^--K P_yK^\mathsf T.
\]

特征分解的主要目标是从 normal equations 构造等价的 `B/y` 虚拟量测，以复用
标准 Kalman 公式；`HtH` 降秩时 `B` 行数减少只是自然结果。LLT 分解作用于始终含
单位阵的 \(P_y\)，不是作用于可能降秩的 `HtH`。

> **来源** · 2026-09-26 · ROVIO LiDAR 信息矩阵与虚拟量测 · chat: `7c120f61-7bc0-40f4-b8c4-d74a774d3ad0` · workspace: `rob_slam-research`

### Map Tracking 虚拟 database 帧

Map Tracking 不执行 Global VPR，也不从 summary map 选一张已有 database
图像。它使用当前 VIO 位姿，把可能可见的 summary-map 3D landmark 投影到
当前相机，现场构造一个只含 keypoint、score、descriptor 和 landmark ID 的
虚拟 `FrameFromSummaryMap`，再与当前 query 图像做点特征匹配和几何验证。

#### 1. 当前位姿与入口

`localizer-flow.cc` 用最近一次 ROVIO 状态构造 IMU 在全局地图系的位姿：

```cpp
T_ItoG = last_rovio_estimate_.T_G_M *
         last_rovio_estimate_.vinode.get_T_M_I();
```

Map Tracking 将其传给 `localizeNFrameMapTracking()`，最终以
`global_localization=false` 调用 loop detector。对 NFrame 中每个相机：

```cpp
const aslam::Transformation T_ItoC =
    n_camera.get_T_C_B(frame_idx);
const aslam::Transformation T_CtoG =
    T_ItoG * T_ItoC.inverse();

extractFrameFromSummaryMap(
    T_CtoG, n_camera.getCamera(frame_idx),
    /*max_distance=*/3.0, &frames_from_summary_map[0]);
```

这里 `T_CtoG` 表示当前相机在全局地图系中的位姿；候选地标最大三维距离
**硬编码为 3.0 m**，当前没有 YAML/gflag 配置。

#### 2. 地标候选筛选

`extractLandMarkIndicesInFoVAndDistance()` 依次执行：

1. 用 1 m 空间网格获取当前相机附近的地图点候选。
2. 球形距离门控：`‖p_landmark - p_camera‖ <= 3.0 m`。
3. 相机 FoV 门控：地标必须位于相机前方，并在相机模型给出的
   left/right/top/bottom border 内。
4. 地标历史观察方向门控。
5. 地标历史观察距离门控。

观察方向门控的核心逻辑：

```cpp
Eigen::Vector3d d = lm_pos - camera_position_in_G;
Eigen::Vector3d d_xy(d.x(), d.y(), 0.0);
Eigen::Vector3d d_normalized = d_xy.normalized();

double min_diff_dot = std::max(
    std::cos(landmark_observer_yaw_range + 30.0 * DEG2RAD), 0.0);
if (d_normalized.dot(landmark_observer_mean_vec_inG) < min_diff_dot) {
  continue;
}
```

- `d_normalized`：当前相机指向地标的水平单位向量。
- `landmark_observer_mean_vec_inG`：建图时该地标历史观察方向的平均单位向量。
- 允许偏角：
  `current_angle <= min(landmark_observer_yaw_range + 30°, 90°)`。
- `+30°` 是运行时放宽量；`max(cos(...), 0)` 将最大允许偏角限制为 90°，
  防止从地标相反一侧取点。

观察距离门控：

```cpp
double dist_diff =
    std::abs(d.norm() - landmark_observer_mean_dist);
if (dist_diff > landmark_observer_dist_range + 2.0) {
  continue;
}
```

即当前观察距离必须落在历史距离范围额外放宽 **2.0 m** 后的区间内。
`mean_yaw/yaw_range/mean_dist/dist_range` 是建 summary map 时按每个地标的
历史观测统计并存入地图的数据，不是 localization YAML 参数。LiDAR 补充地标
创建时的初值为 `yaw_range=45°`、`dist_range=2.0 m`。

#### 3. 投影并构造虚拟帧

通过筛选的 3D landmark 使用当前 `T_CtoG` 和真实相机模型投影：

```cpp
const Eigen::Vector3d lm_pos_C =
    T_CtoG.inverse().transform(lm_pos_G);
const aslam::ProjectionResult result =
    camera.project3(lm_pos_C, &keypoint);
if (!result.isKeypointVisible()) {
  continue;
}
```

投影后还会：

- 以 **8×8 pixel** 为遮挡网格，每格只保留深度最近的 landmark。
- 候选超过 **800** 时，按地图中的 point score 取最高 800 个。
- 将投影像素写入 `keypoints`，从 summary map 复制对应 `scores`、
  `descriptors` 和 `landmark_ids`。
- 虚拟帧没有真实 database 图像；`observer_image` 只用于调试显示。

随后 query 图像特征与虚拟帧 descriptor 做 SP/SG（或 LightGlue）匹配，
再由 `computeAbsoluteTransformFromFrameMatches()` 做几何验证。Map Tracking
失败可能发生在候选为空、点匹配不足或几何验证失败中的任一阶段。

#### 4. 与 Global NMS 的区别

`lc_dl_image_descriptor_match_combined_nms_radius=5` 只作用于 Global VPR 的
observer 帧候选去重，不参与上述虚拟帧生成。其组合距离为：

```text
combined_distance =
    translation_distance_m + angle_distance_deg / 45 * 3
```

组合距离小于 5 的低分候选会被 NMS 丢弃；Map Tracking 不运行 image
descriptor brutal-force，因此也不使用该阈值。

**关键路径**

- `applications/rovioli/src/localizer-flow.cc`
- `applications/rovioli/src/localizer.cc`
- `algorithms/loopclosure/deep-learning-loop-detector/src/deep_learning_loop_detector.cpp`
- `map-structure/localization-summary-map/src/localization-summary-map-creation.cc`

> **来源** · 2026-10-26 · Map Tracking 虚拟帧 · chat: `未找到` · workspace: `rob_slam-research`

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

---

## Docker

> 常用的权限、目录挂载、RViz 显示和 GPU 选择参数。

### 用 Docker 修复宿主机目录所有权

适用场景：当前账号没有 `sudo`，但属于 `docker` 组；个人目录被错误 UID
拥有，导致 Cursor Server 或普通文件无法写入。Docker 容器默认以 root
运行，通过 bind mount 修改的是宿主机文件；`--rm` 只在命令结束后删除临时
容器，不会撤销文件修改。

先确认当前账号的数字 UID/GID，并查看目录中的所有者分布：

```bash
uid=$(id -u)
gid=$(id -g)
printf 'uid=%s gid=%s\n' "$uid" "$gid"

ls -ldn /nfs/home/maoli /PnP/maoli
find /PnP/maoli -xdev -printf '%U:%G\n' | sort | uniq -c
```

#### `/nfs/home/maoli`：整个个人 home 批量改回当前账号

仅在确认整个目录都是自己的数据后执行：

```bash
uid=$(id -u)
gid=$(id -g)

docker run --rm \
  -v /nfs/home/maoli:/target \
  alpine:3.20 \
  chown -R "$uid:$gid" /target
```

验证：

```bash
ls -ldn /nfs/home/maoli
find /nfs/home/maoli -xdev ! -uid "$(id -u)" -print -quit
```

如果只需修复 Cursor，缩小挂载范围：

```bash
uid=$(id -u)
gid=$(id -g)

docker run --rm \
  -v /nfs/home/maoli/.cursor-server:/target \
  alpine:3.20 \
  chown -R "$uid:$gid" /target
```

#### `/PnP/maoli`：只替换旧 owner，保留 `root` 文件

先从前面的所有者分布中确定错误 owner 的数字 UID。每次 owner 可能不同，
所以不要永久写死旧 UID。以下以旧 UID `7710` 为例：

```bash
uid=$(id -u)
gid=$(id -g)
old_uid=7710

# 只读预览
docker run --rm \
  -v /PnP/maoli:/target:ro \
  alpine:3.20 \
  find /target -xdev -user "$old_uid" -print

# 确认后批量修改；-h 避免跟随符号链接修改链接目标
docker run --rm \
  -v /PnP/maoli:/target \
  alpine:3.20 \
  find /target -xdev -user "$old_uid" \
  -exec chown -h "$uid:$gid" {} +
```

验证旧 UID 已清空：

```bash
find /PnP/maoli -xdev -uid "$old_uid" -print
```

注意：

- `/PnP/maoli` 中原属 `root:root` 的文件不会被上述定向命令修改。
- 如果看到 `Operation not permitted`，通常是 NFS 启用了 `root_squash`；
  `--privileged` 也无法绕过 NFS 服务端权限，需要管理员在存储端修复。
- 如果目录出现他人的私钥或明显的个人数据，不要直接接管，先确认目录是否
  分配错误。

> **来源** · 2026-10-23 · Docker 批量修复目录所有权 · chat: `67b43b08-476d-4baf-b811-a97c142059a3` · workspace: `notebook`

### docker run

更灵活的选择是：创建容器时用 `--gpus all` 暴露所有 GPU，执行训练或推理任务时再用 `CUDA_VISIBLE_DEVICES` 指定 GPU。这样长期容器切换 GPU 时不需要删除或重建。

示例：以宿主机 UID/GID 运行、挂载 `/data/maoli`、支持 RViz，并暴露所有 GPU：

```bash
docker run -it \
  --gpus all \
  --user 1000:1000 \
  -e HOME=/home/maoli \
  -e DISPLAY="$DISPLAY" \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e QT_X11_NO_MITSHM=1 \
  -v /data/maoli:/data/maoli \
  -v "$XAUTHORITY:/tmp/.docker.xauth:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -w /data/maoli \
  --net=host \
  --name=humble \
  superodom-ros2:latest \
  /bin/bash
```

- 权限：`--user 1000:1000`、`HOME=/home/maoli`
- 挂载：宿主机和容器都使用 `/data/maoli`
- 可视化：传入 `DISPLAY`、`XAUTHORITY`，并挂载 X11 socket
- GPU：`--gpus all` 让容器能够看到宿主机全部 GPU

运行具体任务时再选择 GPU：

```bash
# 使用物理 GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py

# 使用物理 GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py

# 使用物理 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

`CUDA_VISIBLE_DEVICES` 只限制当前进程及其子进程，不改变容器配置。指定物理 GPU 1 后，该进程通常会将它重新编号为逻辑 `cuda:0`。

### docker exec

进入现有 `humble` 容器：

```bash
docker exec -it \
  -u 1000:1000 \
  -e HOME=/home/maoli \
  -w /data/maoli \
  humble /bin/bash
```

`docker exec` 只需指定用户、`HOME` 和工作目录；目录、GPU 和 X11 应在创建容器的 `docker run` 阶段配置。

> **来源** · 2026-09-30 · Docker run 与 exec 常用参数 · chat: `aa6a397c-f896-4a27-801a-4c15483949d4` · workspace: `SuperOdom`

---

## C++语法规则

> 记录容易误读的 C++ 语法、标准库接口及其使用语义。

### 原子比较交换 compare_exchange_strong

```cpp
bool expected = false;
if (flag.compare_exchange_strong(expected, true)) {
  // Only this caller changes flag from false to true.
}
```

- 可按 **compare + exchange** 记忆：先把原子变量当前值与 `expected` 比较，
  相等时才写入第二个参数 `true`。这种操作通常称为 CAS
  （Compare-And-Swap）。
- 第一个参数 `expected` 是输入/输出参数：
  - 当前值为 `false`：比较成功，将原子变量写成 `true`，返回 `true`；
    `expected` 保持 `false`。
  - 当前值为 `true`：比较失败，不写入原子变量，返回 `false`；
    同时将实际当前值写回 `expected`，所以它变成 `true`。
- `strong` 表示不会出现 `weak` 版本允许的“伪失败”：值明明相等却返回失败。
  因此 `strong` 常用于只尝试一次的 `if`；`weak` 常放在重试循环中。
- 它相当于线程安全地把“检查是否为 `false`”和“设置为 `true`”合并成一个
  不可分割的操作，可用于保证多个调用者中只有一个取得执行权。

> **来源** · 2026-10-21 · C++ 原子比较交换语义 · chat: `未找到` · workspace: `rob_slam-research`
