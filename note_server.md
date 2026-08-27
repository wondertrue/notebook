# Notebook

个人学习笔记。当前仅维护本文件；后续会跨 workspace 补充各模块内容。

## Contents

- [深度学习基础知识](#深度学习基础知识)
- [特征点提取和匹配](#特征点提取和匹配)
- [VLN](#vln)
- [3D-GS](#3d-gs)

---

## 深度学习基础知识

> 基础概念、训练流程（step/epoch、蒸馏、验证）、以 MobileNet-SP 蒸馏为实例。

### 基础概念

**Step 与 Epoch**

- **1 step** = 1 次梯度更新 = 处理 **1 个 batch**（如 batch_size=16 即 16 张图），**不是**一遍全量训练集。
- **1 epoch** ≈ 遍历训练集一次 ≈ `训练集张数 / batch_size` 个 step。
- 公式：`epoch 数 = max_steps × batch_size / 训练集张数`（例：100k×16/31970≈50 epoch）。

**知识蒸馏（稠密监督）**

- Teacher（冻结）对 **clean 图** 产出稠密 target；Student 对输入（可增强）产出预测。
- 在 `H/8 × W/8` 特征图上 **逐 cell 对齐** 比较，不是整张图的单一分类 logits。
- 训练目标：Student 在退化输入上仍复现 Teacher 在理想输入上的 detector 分布与 descriptor 向量。

### 网络结构

（本对话未展开；见 `rob_slam-research` 项目 `train/mobilenet_sp/model.py`。）

### 训练与优化

**数据增强（仅 student 输入）**

- 光度增强：亮度、对比度、高斯噪声、阴影、运动模糊；顺序随机；**无几何变换**（像素空间对应不变）。
- `teacher_image` = resize 后的 clean 图；`student_image` = 增强版（同源）。
- **验证/评估**：`augment=False`，`teacher_image` 与 `student_image` 均为 clean；验证时 student forward 也用 clean（与训练不同）。

**蒸馏损失（分开定义，再组合）**

| 子损失 | 形式 | 说明 |
|--------|------|------|
| Detector | 软交叉熵 | student logits vs teacher `dense_scores`（65 维 softmax，含 dustbin） |
| Descriptor | 稠密 L2 / 2 | 单位向量时等价于 `1 - cosine` |
| 总损失 | `HFNetUncertaintyLoss` | 两项分别算标量，再用可学习 `logvar` 加权合成；detector 项额外 ×2 |

**Validation 与 checkpoint**

- 按 **route group** 固定 hold-out，非 k-fold cross validation。
- 训练时定期在 val 上算 `det_loss` / `desc_loss` / `mean_cosine` 并 **写入 checkpoint**，但 **无** 自动 `best.pth`、early stopping 或 val 驱动 LR。
- LR 按固定步数衰减（如 60k→1e-4，80k→1e-5），选最优 step 需 **人工对比** 各 `step_*.pth` 中的 `validation` 字段。

**收敛与扩数据（`distill_full_v1`，100k 步）**

- Validation（1024 张分层采样）：100k 时 det≈0.338，desc≈0.048，mean_cosine≈0.952；**75k 后进入平台期**，95k 略优于 100k。
- 加训练数据后：同样 100k 步 → 每张图被看到的 **epoch 数减少**（数据翻倍约减半）；若要保持同等覆盖需 **同比增加 max_steps**，或接受更少 epoch。
- 加 store 数据主要改善 **held-out test 泛化**；同一条 val 上的 dense 指标不一定更高，应以 test 关键点/匹配指标为准。

### 笔记

- 实现参考：`/PnP/maoli/git/rob_slam-research/train/mobilenet_sp/`（`data.py` / `losses.py` / `train.py`）。
- 归档文档：`.ai/plans/zh-CN/2026-08-26-mobilenet-sp-distill-onnx.md`。
- 远程 Agent 笔记本路径：`/PnP/maoli/git/notebook/note_server.md`（须放在服务器可写路径，非本机独占目录）。

> **来源** · 2026-08-27 · MobileNet-SP 蒸馏训练基础 · chat: `d1cc100e-0665-49ab-aca9-c4cf9d92fe9f` · workspace: `maoli`

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

### 方法

### 笔记

---

## 3D-GS

3D Gaussian Splatting。

> 待补充：表示与渲染、训练流程、变体与应用。

### 表示与渲染

### 训练

### 笔记
