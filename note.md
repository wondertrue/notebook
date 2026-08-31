# Notebook

个人学习笔记。当前仅维护本文件；后续会跨 workspace 补充各模块内容。

## Contents

- [深度学习基础知识](#深度学习基础知识)
- [特征点提取和匹配](#特征点提取和匹配)
- [VLN](#vln)
- [3D-GS](#3d-gs)
- [建图和定位](#建图和定位)

---

## 深度学习基础知识

> 待补充：基础概念、网络结构、训练要点、常用资源。

### 基础概念

### 网络结构

### 训练与优化

### 笔记

---

## 特征点提取和匹配

> 待补充：检测器、描述子、匹配策略、几何验证。

### 检测与描述

### 匹配与几何验证

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
