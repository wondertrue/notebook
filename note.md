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
