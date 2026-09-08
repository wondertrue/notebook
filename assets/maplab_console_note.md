# Maplab Console 笔记

本文记录 `maplab_console` 后处理过程中涉及的 VIMap 数据结构，以及
`kfh` 删除非关键帧 vertex 的执行链路。回环检测与 landmark 合并暂不纳入。

## 关键数据结构

### VIMap、Mission、Vertex 与 Edge

- `VIMap` 持有 `PoseGraph`、missions、mission baseframes、`LandmarkIndex`
  和传感器管理器；`landmark_index` 是每个 `VIMap` 实例的成员，不是全局变量。
- 一个 mission 是一条具有固定 backbone 类型的轨迹。`getNextVertex()` 和
  `getPreviousVertex()` 只沿该 backbone 遍历：
  - `kViwls` mission 使用 `ViwlsEdge`；
  - `kOdometry` mission 使用 odometry edge；
  - `kWheelOdometry` mission 使用 wheel-odometry edge。
- `Vertex` 表示一个时刻的状态及其 `VisualNFrame`。时间上连续的 vertex 由
  backbone edge 连接。
- `VisualFrameIdentifier = (vertex_id, frame_index)`：
  `frame_index` 是同一 `VisualNFrame` 内的相机槽位，不是轨迹图像序号。
- `KeypointIdentifier = (vertex_id, frame_index, keypoint_index)`：
  唯一定位某个 vertex、某路相机图像中的一个特征点。

### Vertex、VisualNFrame 与像素

每个 `Vertex` 带一个 `VisualNFrame`（该时刻各相机的图像帧）：

```cpp
// vertex.h
aslam::VisualNFrame::Ptr n_frame_;
std::vector<LandmarkIdList> observed_landmark_ids_;
```

- `n_frame_`：该时刻的 N 相机图像；`frame_index` 对应其中一路 `aslam::VisualFrame`。
- `observed_landmark_ids_[frame_idx]`：与该相机 `VisualFrame` 的 keypoint **列对齐**，
  第 `i` 个点对应哪个 `LandmarkId`（可 invalid，表示尚未关联 3D 点）。
- 像素坐标不在 `Landmark` 里，而在 `VisualFrame` 的测量 channel 里。

| 数据 | 位置 | 含义 |
|------|------|------|
| `keypointMeasurements`（`Eigen::Matrix2Xd`，2×N） | `VisualFrame` | 该相机图像上每个 keypoint 的 `(u, v)` |
| 描述子、score、scale、uncertainty | 同一 `VisualFrame` | 与像素列对齐 |
| `observed_landmark_ids_[frame_idx][keypoint_index]` | `Vertex` | 第几个点对应哪个 `LandmarkId` |
| `Landmark.observations_` | `Landmark` | 反向：哪些 `(vertex, frame, keypoint_index)` 看到我 |
| `Landmark.p_B` | `Landmark` | 3D 位置，没有像素 |

因此：查某个观测的像素，走
`(vertex, frame_index, keypoint_index)` → `n_frame_->getFrame(frame_index)` →
`keypointMeasurements.col(keypoint_index)`；查 3D 点则走 `LandmarkId` → store
vertex 上的 `Landmark`。

> **来源** · 2026-09-08 · Vertex VisualNFrame 与像素 · chat: `51785407-f840-4bff-bf7a-47dc01e3a042` · workspace: `rob_slam-research`

### Landmark 的三类关联

同一个 landmark 只有一个实体，但可以被多个 vertex 观测：

```text
LandmarkIndex:
  landmark_id -> store_vertex_id

store_vertex.landmarks_:
  Landmark { id, p_B, observations, quality, ... }

Landmark.observations_:
  (observer_vertex_id, frame_index, keypoint_index) ...

observer_vertex:
  keypoint -> observed_landmark_id
```

- **LandmarkStore**：某个 vertex 保存 landmark 实体；每个 landmark 恰好有一个
  store vertex。
- **LandmarkIndex**：`unordered_map<LandmarkId, VertexId>`，用于从
  `landmark_id` 快速找到唯一 store vertex，避免遍历所有 vertex。
- **`Landmark::observations_`**：landmark 到所有观测 keypoint 的反向引用。
  多个条目可以来自不同 vertex，也可以来自同一 vertex 的不同相机。
- **Vertex 的 `observed_landmark_id`**：keypoint 到 landmark 的正向引用。

删除 landmark 时必须保持三者一致：

1. 根据 `observations_` 将各 observer vertex 的正向引用置为 invalid；
2. 从 `LandmarkIndex` 删除 `landmark_id -> store_vertex_id`；
3. 从 store vertex 的 `LandmarkStore` 删除 landmark 实体。

### Landmark 坐标

- `Landmark` 实际保存 `p_B`：landmark 相对其 store vertex 的 body/IMU 坐标。
- 全局坐标 `p_G_fi` 不直接存储，而是按需通过
  `store vertex pose + mission baseframe` 计算。
- 搬迁 store vertex 时，必须先保留原全局位置，再计算 landmark 在新 store
  vertex body 系下的 `p_B`。因此不能只复制旧 `p_B`。

## KFH 删除非关键帧

### 入口与整体调用链

`keyframeMapBasedOnHeuristics()` 先调用
`selectKeyframesBasedOnHeuristics()` 选择保留的 keyframe，再从以下入口删除
相邻 keyframe 之间的 vertex：

```text
keyframeMapBasedOnHeuristics
  -> removeVerticesBetweenKeyframes
     -> removeAllVerticesBetweenTwoVertices
        -> VIMap::mergeNeighboringVertices
```

`removeVerticesBetweenKeyframes(keyframe_ids, map)` 的职责：

1. 校验所有 keyframe 属于同一 mission；
2. 遍历每一对相邻保留 keyframe；
3. 删除两者之间的非关键帧，并累计删除数量；
4. 不删除区间两端的保留 keyframe。

### removeAllVerticesBetweenTwoVertices

假设要保留 `A` 和 `D`，删除中间的 `B`、`C`：

```text
A -> B -> C -> D
```

函数先沿 `getNextVertex()` 从 A 走到 D，检查区间内每一段是否完整具有
Viwls、Odometry、WheelOdometry 三类边。某类 edge 只要缺一段，就把对应
`merge_*_edges` 设为 false，整个区间不再合并该类 edge，避免凭空插值约束。

然后反复执行：

```text
mergeNeighboringVertices(A, B)
mergeNeighboringVertices(A, C)
```

每次删除 A 后面的相邻 vertex，因此最终剩下 `A -> D`。

### mergeNeighboringVertices 的处理顺序

#### 1. 搬迁 LandmarkStore

`moveLandmarksToOtherVertex(vertex_to_merge, merge_into_vertex)`：

1. 遍历待删 vertex 存储的所有 landmark；
2. 由旧 store 关系计算每个 landmark 的全局位置；
3. 转换为相对保留 vertex 的新 `p_B`；
4. 将 landmark 实体加入保留 vertex 的 `LandmarkStore`；
5. 更新 `LandmarkIndex` 指向新的 store vertex；
6. 从待删 vertex 的 `LandmarkStore` 移除旧实体。

这是“landmark 实体搬家”，不是合并两个 landmark，landmark ID 不变。

#### 2. 清理并合并 Edge

- 收集待删 vertex 的所有 incoming/outgoing edges。
- 仅保留允许合并的 Viwls、Odometry、WheelOdometry edges；其他边直接删除。
- 对允许合并的 backbone edge，将：

```text
A -> B -> C
```

合成为：

```text
A ------> C
```

若 B 是末尾 vertex，没有 `B -> C`，则只删除 `A -> B`。
合并后检查 mission 对应的 backbone 仍保持唯一、连续。

#### 3. 清理观测与孤立 Landmark

遍历待删 vertex 各相机帧的所有 `observed_landmark_id`：

- 若 landmark 的 observer vertex 数量为 1，说明待删 vertex 是唯一观测者，
  调用 `removeLandmark()` 同步清理正向引用、反向引用、全局索引和实体。
- 若还有其他 observer vertex，则保留 landmark，仅调用
  `removeAllObservationsOfVertex(vertex_to_merge)` 删除待删 vertex 的反向观测。
- `deleted_landmarks` 防止同一 landmark 被待删 vertex 的多个相机帧观测时重复删除。

#### 4. 删除 Vertex

完成 landmark 搬迁、edge 重连和观测清理后，调用
`posegraph.removeVertex(vertex_to_merge)`。此时不再有指向该 vertex 的有效
landmark 或 edge 关系。

## 核心理解

- KFH 删除非关键帧不是简单 erase，而是“搬 landmark 实体、重连 backbone、
  清理观测、删除孤立 landmark、最后删 vertex”。
- store vertex 和 observer vertex 是两个概念：前者唯一，后者可以有多个。
- `LandmarkIndex` 负责定位实体；`observations_` 负责表达多视图观测。
- `frame_index` 表示 NFrame 内的相机槽位；轨迹时间顺序由 vertex 和 backbone
  edge 表达。
- 像素 `(u, v)` 在 `VisualFrame` 里，不在 `Landmark` 里；`Landmark` 只存 3D
  `p_B` 和观测 backlink。

> **来源** · 2026-09-08 · Maplab Console 数据结构与 KFH · chat: `51785407-f840-4bff-bf7a-47dc01e3a042` · workspace: `rob_slam-research`
