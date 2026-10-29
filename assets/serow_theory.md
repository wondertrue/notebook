# 基于 Go2 MCAP 的 SEROW 足式里程计：输入、观测与模型

> 本文面向第一次学习足式里程计的读者。结论以当前 workspace
> `/data/maoli/git/legged_odom/serow_ws` 的 Go2 slope MCAP、配置和源码为准，
> 不把 SEROW 的可选能力误写成当前 demo 已启用的能力。

## 1. 先用一句话理解

当前 Go2 demo 做的事情是：

1. IMU 高频积分，预测机身姿态、速度和位置；
2. 关节角经过机器人运动学模型，算出每只脚相对机身的位置和速度；
3. 足端力判断哪些脚更可能踩在地上；
4. 假设接触脚相对地面近似静止，反推出机身速度，修正 IMU 积分漂移；
5. 再结合足端力、机器人动力学和 CoM 运动学，估计质心位置、速度及外力。

它本质上是 **IMU 惯性导航 + 接触约束 + 腿部正运动学**。足式里程计不是直接
“从脚力算位姿”，脚力主要告诉估计器“哪只脚可信”；真正提供机身运动约束的是
接触脚的运动学。

---

## 2. 当前数据与运行配置

### 2.1 MCAP 的实际内容

输入文件：

```text
src/serow/evaluation/mujoco_test/quadruped/mujoco_data/slope/go2_data.mcap
```

MCAP 实测信息：

- 通道：`/robot_state`
- 编码：JSON
- schema：`QuadrupedState`
- 消息数：18,092
- 时间范围：2.000 s 到 38.182 s，约 500 Hz

每帧顶层字段：

```text
timestamp
imu
joint_states
feet_forces
feet_contact
base_ground_truth
```

12 个关节为每条腿的 `Hip / Thigh / Calf`：

```text
FL_Hip  FL_Thigh  FL_Calf
FR_Hip  FR_Thigh  FR_Calf
RL_Hip  RL_Thigh  RL_Calf
RR_Hip  RR_Thigh  RR_Calf
```

### 2.2 哪些字段真正进入当前滤波器

| MCAP 字段 | 当前 demo 如何处理 | 是否影响估计 |
|---|---|---|
| `timestamp` | 作为 IMU、关节、足端力的统一时间戳 | 是 |
| `imu.linear_acceleration` | 三轴加速度计读数，单位 m/s² | 是 |
| `imu.angular_velocity` | 三轴陀螺仪读数，单位 rad/s | 是 |
| 12 个关节 `position` | 映射成 SEROW 的 `*_joint` 名称 | 是 |
| 12 个关节 `velocity` | 虽然被解析，但当前配置 `estimate_joint_velocity=true`，实际用关节角微分重算 | 否 |
| 12 个关节 `effort` | 被解析，但当前 `estimate_contact_wrench=false` | 否 |
| `feet_forces.{FL,FR,RL,RR}` | 作为四足局部坐标系中的三维接触力 | 是 |
| `feet_contact` | demo 没有读取 | 否 |
| `base_ground_truth` | 传入 `BasePoseGroundTruth`，只用于日志/评估 | **不参与滤波** |

当前调用明确没有外部视觉/LiDAR odometry，也没有外部接触概率：

```cpp
estimator.filter(
    imu, joints,
    force_torque,
    std::nullopt,   // no external odometry
    std::nullopt,   // no external contact probabilities
    base_pose_gt);  // logging/evaluation only
```

对应代码：

- MCAP 读取与字段映射：
  `src/serow/evaluation/mujoco_test/quadruped/go2_mujoco_test.cpp:122-238`
- 滤波主流程：
  `src/serow/core/src/Serow.cpp:1255-1409`

### 2.3 当前 Go2 配置的关键含义

配置文件：`src/serow/config/go2.json`

```text
base_estimator_type          = left-invariant
point_feet                   = true
imu_rate                     = 500 Hz
joint_rate                   = 500 Hz
force_torque_rate            = 500 Hz
estimate_joint_velocity      = true
estimate_contact_wrench      = false
estimate_contact_status      = true
calibrate_initial_imu_bias   = true
max_imu_calibration_cycles   = 500
enable_terrain_estimation    = true
use_imu_orientation          = false（字段缺省后的默认值）
T_base_to_odom               = null
```

实际效果：

- 前 500 个 IMU 样本假设机器人静止，用约 1 秒估计初始陀螺仪和加速度计 bias；
- Go2 按点足处理，不使用足端力矩和有限面积足底 COP；
- 足端力来自 MCAP，不启用由关节力矩反推接触力的 GMO；
- 基座滤波器为 15 维误差状态 Left-Invariant EKF；
- 没有绝对外部里程计；
- Mahony 姿态供腿部里程计使用，但不会作为一条独立姿态量测再次更新 EKF。

---

## 3. 坐标系与符号

本文采用与代码一致的记号：

- \(W\)：里程计世界坐标系；
- \(B\)：Go2 `base` 坐标系；
- \(F_i\)：第 \(i\) 只脚的坐标系；
- \(R_{WB}\)：把 \(B\) 系向量旋转到 \(W\) 系；
- \(p_{WB}\)：base 原点在 \(W\) 系的位置；
- \(p_{BF_i}\)：足端相对 base 的位置，用 \(B\) 系表达；
- \(v_{WB}\)、\(\omega_{WB}\)：base 在世界系中的线速度和角速度；
- \(q,\dot q\)：12 维关节角和关节角速度；
- \([x]_\times y=x\times y\)：叉乘矩阵；
- \(g_W=[0,0,-9.81]^\mathsf T\)。

源码中的 `base_orientation` 实际表示 \(R_{WB}\)，即：

\[
v_W = R_{WB}v_B.
\]

参考：

- `src/serow/core/src/State.hpp:34-85`
- `src/serow/core/src/Mahony.hpp:82-89`

---

## 4. 从原始输入构造内部观测

这里要区分三个概念：

1. **传感器输入**：IMU、编码器、足端力；
2. **内部观测量**：足端位姿、接触概率、腿部里程计速度、CoM 运动学位置等；
3. **滤波状态输出**：base 位姿/速度、bias、CoM 状态和外力。

### 4.1 IMU 测量模型

陀螺仪模型：

\[
\omega_m = \omega_B + b_g + n_g.
\]

加速度计测的是比力，不是世界系线加速度：

\[
a_m = R_{WB}^{\mathsf T}(a_W-g_W)+b_a+n_a.
\]

因此去 bias 后恢复世界系加速度：

\[
a_W = R_{WB}(a_m-b_a)+g_W.
\]

代码的重力向量是 \(g_W=[0,0,-g]^\mathsf T\)，预测公式见：

- `src/serow/core/src/LeftInvariantEKF.cpp:190-221`
- `src/serow/core/src/Serow.cpp:1068-1084`

Mahony filter 还用“短时间内加速度主要来自重力”的假设，将归一化加速度方向
与估计重力方向对齐，从而约束 roll/pitch：

\[
e_g = \hat a_m \times \hat g_B.
\]

陀螺仪积分提供快速姿态变化，加速度方向提供重力校正。仅靠这两者不能观测绝对
yaw，因为绕重力轴旋转不会改变重力方向。

对应代码：`src/serow/core/src/Mahony.hpp:121-203`。

### 4.2 关节角到足端运动学

#### 4.2.1 输入和输出分别是什么

Go2 每条腿有 3 个转动关节：

```text
Hip → Thigh → Calf → Foot
```

MCAP 一帧给出 12 个关节角。把它们排成向量：

\[
q=
\begin{bmatrix}
q_{FL,H}&q_{FL,T}&q_{FL,C}&\cdots&q_{RR,C}
\end{bmatrix}^{\mathsf T}.
\]

URDF 描述每个关节：

- 绕哪一根轴旋转；
- 父、子连杆之间的固定平移和旋转；
- 连杆的质量、质心和惯量。

因此，给定关节角后，可以沿 `base → hip → thigh → calf → foot` 逐级连乘刚体
变换，得到每只脚相对 base 的位置和姿态。Pinocchio 替代码完成了这部分递归计算。

第 \(i\) 只脚的正运动学写成：

\[
p_{BF_i}=f_i(q),
\]

其中：

- \(q\) 是所有关节角；
- \(f_i(\cdot)\) 是由 URDF 决定的非线性函数；
- \(p_{BF_i}\) 是“从 base 原点指向第 \(i\) 只脚”的向量，用 base 系表示。

虽然公式写了完整 \(q\)，但一只脚通常只受本腿 3 个关节影响。例如
`FL_foot` 对 `RR_*` 关节的偏导为零。

除位置外，代码还计算：

- \(R_{BF_i}\)：足端姿态；
- \(v_{BF_i}\)：仅由关节运动造成的足端相对线速度；
- \(\omega_{BF_i}\)：仅由关节运动造成的足端相对角速度。

这里的“相对”很重要：此时还没有加上 base 自身的平移和旋转。

#### 4.2.2 Jacobian 是什么

Jacobian 不需要先理解成复杂矩阵。它就是“关节角发生一点变化时，足端位置会变化
多少”的局部比例关系：

\[
\delta p_{BF_i}\approx J_{v,i}(q)\,\delta q.
\]

数学上：

\[
J_{v,i}(q)=\frac{\partial f_i(q)}{\partial q}.
\]

如果机器人有 12 个关节，\(J_{v,i}\) 的尺寸是 \(3\times12\)：

- 3 行对应足端位置的 \(x,y,z\)；
- 12 列对应 12 个关节；
- 与该脚无关的关节列通常为零。

把上式两边除以很小的时间间隔，就得到速度关系：

\[
v_{BF_i}=J_{v,i}(q)\dot q,
\qquad
\omega_{BF_i}=J_{\omega,i}(q)\dot q.
\]

直观例子：机器人站立时，如果膝关节收缩，小腿末端会向上、向后移动。
Jacobian 就把“膝关节每秒转多少 rad”换算成“脚相对 base 每秒移动多少 m”。

#### 4.2.3 当前代码如何得到 \(\dot q\)

MCAP 中虽然有每个关节的 `velocity`，但当前配置是：

```json
"estimate_joint_velocity": true,
"smoothing_window": 11
```

所以 demo 不采用 MCAP 自带速度，而是根据连续 11 帧关节角做平滑微分：

\[
\dot q_k \approx \mathrm{DerivativeEstimator}
    (q_{k-M+1:k}).
\]

在 500 Hz 下，相邻样本相隔约 2 ms，11 帧覆盖约 20 ms 的首尾时间跨度。
这样做比简单的一阶差分

\[
\dot q_k\approx\frac{q_k-q_{k-1}}{\Delta t}
\]

更不容易放大编码器噪声，但也会引入一定平滑和延迟。

#### 4.2.4 运动学不确定度

如果关节角噪声 covariance 为 \(\Sigma_q\)，一阶误差传播得到足端位置 covariance：

\[
\Sigma_{p_i}=J_{v,i}\Sigma_qJ_{v,i}^{\mathsf T}.
\]

关节速度噪声同理：

\[
\Sigma_{v_i}=J_{v,i}\Sigma_{\dot q}J_{v,i}^{\mathsf T}.
\]

这说明腿伸得很直、接近运动学奇异位形时，即使关节噪声相同，足端位置或速度的
某些方向也可能变得更不确定。

对应代码：

- 关节速度选择：`src/serow/core/src/Serow.cpp:627-679`
- 足端正运动学：`src/serow/core/src/Serow.cpp:746-796`
- Jacobian 与速度：`src/serow/core/src/RobotKinematics.hpp:242-282,350-480`

### 4.3 足端力到接触概率

#### 4.3.1 为什么需要判断接触

关节编码器始终能算出脚相对 base 的运动，但只有脚踩住地面时，才能用“脚在世界
中近似不动”约束 base：

- 支撑腿：可以提供里程计约束；
- 摆动腿：脚在主动向前摆，不能当作固定地面点；
- 打滑腿：虽然承重，但静止假设已经不完全成立。

当前 SEROW 用足端力判断一只脚有多像“支撑腿”。

#### 4.3.2 先统一力的坐标系

MCAP 的 `feet_forces.FL/FR/RL/RR` 是各自局部足端/传感器系的三维力。不同脚的
坐标轴会随腿姿态改变，不能直接比较其 \(z\) 分量。代码先将其转到 base 系：

\[
f^B_i = R_{BF_i}R_{F_iS_i}f^{S_i}_i.
\]

其中：

- \(f^{S_i}_i\)：传感器系读数；
- \(R_{F_iS_i}\)：传感器系到足端系的安装旋转；
- \(R_{BF_i}\)：足端系到 base 系的当前运动学旋转。

当前 `R_foot_to_force=I`，即 \(R_{F_iS_i}=I\)，所以主要使用当前足端姿态
\(R_{BF_i}\) 做旋转。之后代码把 base 系力的 \(z\) 分量当作竖直支撑力。
这要求 base 的 \(z\) 轴定义和配置正确。

#### 4.3.3 为什么先截断再做中值滤波

对每只脚的竖直力，`ContactDetector` 先执行：

1. 截断到 \([0,mg]\)；
2. 移动中值滤波；
3. 得到 \(\tilde f_{z,i}\)。

截断的含义：

- 小于 0 的值不作为支撑力；
- 超过整机重量 \(mg\) 的单足瞬时尖峰被限制；
- 它是工程保护，不代表真实单足力在任何动态动作中都绝不可能超过 \(mg\)。

中值滤波适合去除孤立尖峰。例如窗口中的读数为：

```text
118 N, 121 N, 900 N, 119 N, 122 N
```

中值约为 `121 N`，不会像均值那样被 `900 N` 明显拉高。

#### 4.3.4 当前“接触概率”的精确定义

设滤波后的竖直力为 \(\tilde f_{z,i}\)，当前连续接触概率为：

\[
c_i =
\begin{cases}
\mathrm{clamp}
\left(
\dfrac{\tilde f_{z,i}}
{\frac{1}{N}\sum_j\tilde f_{z,j}},0,1
\right), & \sum_j\tilde f_{z,j}>0,\\[6pt]
0,&\text{otherwise}.
\end{cases}
\]

二值接触状态为：

\[
s_i = (c_i>0.5).
\]

这里的 \(c_i\) 更准确地说是 **接触置信权重**，不是经过统计标定的概率。
它不要求 \(\sum_i c_i=1\)，多个承重足可以同时得到 1。

四足例子一：

```text
滤波竖直力 = [100, 100, 0, 0] N
四足平均值 = 50 N
接触置信度 = clamp([2, 2, 0, 0]) = [1, 1, 0, 0]
```

这表示两只脚都被认为可靠接触。

例子二：

```text
滤波竖直力 = [80, 20, 0, 0] N
四足平均值 = 25 N
接触置信度 = clamp([3.2, 0.8, 0, 0]) = [1, 0.8, 0, 0]
```

第一只脚比第二只脚获得更高置信度。

后续 `LegOdometry` 还会再归一化一次：

\[
w_i=\frac{c_i}{\sum_jc_j},
\qquad \sum_iw_i=1,
\]

此时的 \(w_i\) 才是多足融合权重。二值状态 \(s_i\) 主要用于接触事件和其他状态
逻辑；腿部里程计融合使用的是连续 \(c_i\)，不是只用 0/1。

#### 4.3.5 足端力还用于什么

足端力不只用于接触判断。变换到世界系后，四足力还会求和得到 Ground Reaction
Force，供后面的 CoM/外力 EKF 使用。因此同一份输入有两种作用：

```text
足端力
  ├─ 竖直分量 → 接触置信度 → 哪些腿约束 base
  └─ 三维向量求和 → GRF → CoM 和外力估计
```

对应代码：

- 中值滤波和截断：`src/serow/core/src/ContactDetector.hpp:46-68`
- 概率与二值状态：`src/serow/core/src/Serow.cpp:887-957`

当前 MCAP 的 `feet_contact` 没有进入这条链路。

### 4.4 接触脚静止约束：足式里程计的核心

#### 4.4.1 先看位置关系

足端世界系位置满足：

\[
p_{WF_i}=p_{WB}+R_{WB}p_{BF_i}.
\]

可以把它读成：

```text
脚的世界位置
= base 的世界位置
+ 从 base 指向脚的向量（旋转到世界系后）
```

注意“脚静止”表示 \(p_{WF_i}\) 在一小段支撑期内近似不变，不表示
\(p_{WF_i}=0\)。

#### 4.4.2 对位置求导为什么会出现三项

对时间求导：

\[
v_{WF_i}
=v_{WB}
+\omega_{WB}\times(R_{WB}p_{BF_i})
+R_{WB}v_{BF_i}.
\]

三项分别是：

1. \(v_{WB}\)：base 整体平移带着脚一起运动；
2. \(\omega_{WB}\times(R_{WB}p_{BF_i})\)：base 自身旋转造成的脚端切向速度；
3. \(R_{WB}v_{BF_i}\)：关节转动造成的脚相对 base 的速度。

第二项来自旋转向量求导公式：

\[
\frac{d}{dt}(Rr)=\omega\times(Rr)+R\dot r.
\]

如果漏掉旋转项，机器人原地转身时会被错误估计出额外平移速度。

#### 4.4.3 从“脚不动”反求 base 速度

如果第 \(i\) 只脚可靠接触地面，采用“足端近似静止”假设：

\[
v_{WF_i}\approx 0.
\]

于是单只接触脚给出的 base 速度观测为：

\[
\hat v^{(i)}_{WB}
=-\omega_{WB}\times(R_{WB}p_{BF_i})
-R_{WB}v_{BF_i}.
\]

这条公式的直觉是：为了让世界中的支撑脚保持不动，base 的速度必须抵消
“base 旋转造成的足端速度”和“关节运动造成的足端速度”。

最简单的直线例子：忽略旋转且 \(R_{WB}=I\)。机器人向前走时，支撑脚相对 base
会向后移动。若编码器算出：

\[
v_{BF_i}=
\begin{bmatrix}-0.5&0&0\end{bmatrix}^{\mathsf T}\ \mathrm{m/s},
\]

那么：

\[
\hat v_{WB}^{(i)}=-v_{BF_i}
=
\begin{bmatrix}0.5&0&0\end{bmatrix}^{\mathsf T}\ \mathrm{m/s}.
\]

也就是 base 正以 0.5 m/s 向前运动。

若机器人原地绕 \(z\) 轴转动，即使关节角暂时不变，
\(\omega\times(Rp_{BF_i})\) 也不为零；base 平移项必须和它一起考虑，才能区分
“转身造成的脚端切向运动”和“真正的平移”。

#### 4.4.4 多只支撑脚怎样融合

多足同时接触时，先根据上一节的接触置信度得到：

\[
w_i = \frac{c_i}{\sum_j c_j},
\qquad
\hat v^{leg}_{WB}=\sum_i w_i\hat v^{(i)}_{WB}.
\]

例如两只脚的权重分别是 0.7 和 0.3，它们给出的速度分别为 0.50 m/s 和
0.60 m/s，则融合结果为：

\[
0.7\times0.50+0.3\times0.60=0.53\ \mathrm{m/s}.
\]

这样可以降低单腿编码器噪声的影响，但不能自动消除共同误差，例如所有支撑脚都
在同一块光滑地面上一起滑动。

#### 4.4.5 当前源码中这个速度的准确角色

当前 `LegOdometry.cpp` 先计算上述加权速度，变量名是：

```cpp
base_linear_velocity_kinematic
```

但它 **不是直接送给 EKF 的最终速度量测**。它首先用于计算每只脚的瞬时支点
IMP。随后代码根据接触锚点得到一条 base 位置序列，再对该位置序列做平滑微分；
这个微分结果 `base_linear_velocity_` 才会写入 `KinematicMeasurement` 并用于
EKF 更新。完整过程见下一节。

另一个源码细节：物理公式中的 \(\omega_{WB}\) 应是去 bias 的角速度；当前
`computeLegOdometry()` 实际传入的是：

```cpp
Rwb * imu.angular_velocity
```

也就是 Mahony 姿态旋转后的当前 gyro 读数，此处没有再次显式减去 EKF 的 gyro
bias。EKF 自身的 IMU 预测会减 bias，但这条腿部运动学支路按当前代码并未减。

对应实现：

- 速度约束与多足融合：`src/serow/core/src/LegOdometry.cpp:121-143`
- 传入世界系 gyro：`src/serow/core/src/Serow.cpp:825-835`

### 4.5 接触锚点与 base 位置

#### 4.5.1 什么是接触锚点

当一只脚开始稳定支撑时，可以把它当前的世界位置记作一个短期锚点：

\[
a_i\approx p_{WF_i}.
\]

在理想无滑动支撑期内，\(a_i\) 不变。已知锚点和脚相对 base 的位置，就能反求
base 位置：

\[
\hat p^{(i)}_{WB}=p_{WF_i}-R_{WB}p_{BF_i}.
\]

它与上一节的速度公式是同一个刚体关系：

- 位置层面：固定脚世界位置，反求 base 位置；
- 速度层面：固定脚世界速度为零，反求 base 速度。

一维直观例子：

```text
支撑脚世界坐标 a_i = 0.30 m
脚相对 base 坐标 p_BF = -0.20 m
姿态 R = 1
```

则：

\[
p_{WB}=0.30-(-0.20)=0.50\ \mathrm{m}.
\]

#### 4.5.2 多足位置候选

每只脚都能给出一个 base 位置候选：

\[
p^{(i)}_{candidate}=a_i-R_{WB}p_{BF_i}.
\]

代码用连续接触权重融合：

\[
\hat p^{leg}_{WB}=\sum_i w_i\hat p^{(i)}_{WB}.
\]

支撑力大的脚权重更高；摆动脚的权重接近零。若没有任何可靠接触，
`LegOdometry` 不会用足端候选更新位置，此时 base 主要依赖 IMU 预测。

源码中的 `feet_position_prev_` 可以理解为递推维护的足端世界参考位置，但它并非
一个永远冻结不动的地图 landmark。每轮融合后，代码还会调整各足参考位置，使它们
与融合后的 base 位置和当前运动学关系一致。因此它是局部递推接触锚点，不是全局
地图点。

#### 4.5.3 为什么还需要 Instantaneous Moment Pivot

真实接触不一定严格发生在足端 frame 原点：

- 平足可能绕脚跟或脚尖转动；
- 有限面积足底的受力中心会移动；
- 即使点足模型，数值和模型误差也可能使瞬时支点偏离 frame 原点。

SEROW 用 \(\rho_i\) 表示足端局部坐标系中的瞬时支点偏移，并求解：

\[
A_i\rho_i=b_i,
\]

\[
A_i=
\frac{1}{\Delta t^2}I
-\alpha_1[\omega_{F_i}]_\times^2,
\]

\[
b_i=
\frac{1}{\Delta t^2}\rho_{i,k-1}
+\alpha_1[\omega_{F_i}]_\times R_{WF_i}^{\mathsf T}v_{F_i}.
\]

初学时可以先这样理解各项：

- \(\rho_i\)：要估计的局部支点偏移；
- \(\rho_{i,k-1}/\Delta t^2\)：倾向于让支点随时间平滑，不要突然跳动；
- \(\omega_{F_i}\) 和 \(v_{F_i}\)：利用脚的瞬时运动判断绕哪个点转动最合理；
- \(\alpha_1\)：运动学约束的权重。

当前 \(\alpha_1=\texttt{tau\_0}=5\times10^{-4}\)。Go2 配置
`point_feet=true`，所以不使用足端力矩项。对于平足，源码还可以加入由接触力、
力矩和传感器偏置形成的 \(\alpha_3=\texttt{tau\_1}\) 项，以更好地定位实际支点。

估计出 \(\rho_i\) 后，代码根据足端当前和上一时刻的姿态变化修正世界锚点：

\[
a_i\leftarrow
a_i-R_{WF_i,k}\rho_i+R_{WF_i,k-1}\rho_i.
\]

若 \(\rho_i=0\)，该修正自然消失，退化为“接触点就是足端 frame 原点”。

#### 4.5.4 最终送入 EKF 的腿部速度

代码先得到接触锚点融合位置 \(\hat p^{leg}_{WB,k}\)，再用平滑微分器计算：

\[
\hat v^{leg}_{WB,k}
\approx
\mathrm{DerivativeEstimator}
\left(
\hat p^{leg}_{WB,k-M+1:k}
\right).
\]

因此当前实现的实际数据流是：

```text
关节角 + 姿态 + 接触权重
  → 各脚 base 位置候选
  → 接触锚点加权的 base 位置
  → 对位置序列平滑微分
  → 最终腿部 base 速度量测
  → Left-Invariant EKF 速度更新
```

这也解释了为什么位置锚点质量会影响最终速度：锚点因打滑突然移动时，位置序列的
微分会出现错误速度。

#### 4.5.5 腿部位置是否直接修正 EKF 的 base 位置

当前 Left-Invariant EKF 没有直接用
`kin.base_position` 做一条三维位置更新。它主要使用：

1. `kin.base_linear_velocity` 做速度更新；
2. 接触点建立的 terrain elevation 对 base 高度做更新；
3. IMU 模型继续传播 base 位置。

所以不能把内部的 `LegOdometry::base_position_` 和最终
`State::base_position` 当成同一个变量。前者主要用于生成腿部速度观测，后者是
EKF 状态。

对应实现：

- IMP：`src/serow/core/src/LegOdometry.cpp:66-106`
- 锚点、位置候选与融合：`src/serow/core/src/LegOdometry.cpp:145-188`
- 位置微分得到最终速度：`src/serow/core/src/LegOdometry.cpp:190-216`
- 写入内部观测：`src/serow/core/src/Serow.cpp:836-842`
- EKF 使用速度而非腿部位置：`src/serow/core/src/LeftInvariantEKF.cpp:484-493`

---

## 5. Base 的 Left-Invariant EKF

### 5.1 状态

当前名义状态可写为：

\[
x=(R_{WB},v_{WB},p_{WB},b_g,b_a).
\]

误差状态为 15 维：

\[
\delta x=
\begin{bmatrix}
\delta\theta&
\delta v&
\delta p&
\delta b_g&
\delta b_a
\end{bmatrix}^{\mathsf T}.
\]

对应代码：
`src/serow/core/src/LeftInvariantEKF.cpp:31-77`。

“Left-Invariant”主要描述姿态、速度、位置误差在 Lie group 上的定义与回缩方式，
不是说物理运动方程变了。其状态修正为：

\[
R^+=R\operatorname{Exp}(\delta\theta),
\]

\[
v^+=v+R\delta v,\qquad
p^+=p+R\delta p.
\]

对应代码：
`src/serow/core/src/LeftInvariantEKF.cpp:388-452`。

### 5.2 IMU 预测模型

去 bias 后：

\[
\bar\omega=\omega_m-b_g,\qquad
\bar a=a_m-b_a.
\]

离散预测为：

\[
R_{k+1}=R_k\operatorname{Exp}(\bar\omega_k\Delta t),
\]

\[
v_{k+1}=v_k+(R_k\bar a_k+g_W)\Delta t,
\]

\[
p_{k+1}=p_k+v_k\Delta t+
\frac{1}{2}(R_k\bar a_k+g_W)\Delta t^2.
\]

协方差传播：

\[
P_{k+1}=\Phi P_k\Phi^\mathsf T+Q_c\Delta t,
\]

\[
\Phi\approx I+A_c\Delta t+\frac12A_c^2\Delta t^2.
\]

对应代码：
`src/serow/core/src/LeftInvariantEKF.cpp:150-221`。

### 5.3 腿部速度更新

量测：

\[
y_v=\hat v^{leg}_{WB}.
\]

创新先转到当前 body 切空间：

\[
z_v=R_{WB}^{\mathsf T}
\left(y_v-\hat v_{WB}\right).
\]

线性化观测矩阵只直接选择速度误差：

\[
H_v=
\begin{bmatrix}
0&I&0&0&0
\end{bmatrix}.
\]

标准 EKF 更新：

\[
S=H_vPH_v^\mathsf T+N_v,
\qquad
K=PH_v^\mathsf TS^{-1},
\]

\[
\delta x=Kz_v.
\]

腿部观测虽然直接测速度，但由于 \(P\) 中存在状态交叉协方差，也会间接修正姿态、
位置和 IMU bias。

对应代码：
`src/serow/core/src/LeftInvariantEKF.cpp:650-687`。

### 5.4 Terrain 高度更新

接触足给出世界点：

\[
p_{WF_i}=p_{WB}+R_{WB}p_{BF_i}.
\]

局部 elevation map 在相同 \(x,y\) 查询高度 \(h(x,y)\)，构造竖直创新：

\[
z_h=h(p_{WF_i,x},p_{WF_i,y})-p_{WF_i,z}.
\]

当前实现只让这条量测直接修正 base 高度方向，避免地形高度与姿态、水平位置之间
形成不稳定反馈。

对应代码：
`src/serow/core/src/LeftInvariantEKF.cpp:300-385,500-592`。

要注意：当前 elevation map 本身也是由接触脚逐步建立的，不是外部已知地图。
因此它主要增强局部地面高度一致性，不等价于 GPS、视觉定位或已知地图提供的绝对
位置约束。

---

## 6. CoM 与外力模型

Base 里程计之后还有一个独立的 9 维 CoM EKF：

\[
x_c=
\begin{bmatrix}
c_W&\dot c_W&f_{ext}
\end{bmatrix}^{\mathsf T},
\]

其中：

- \(c_W\)：质心世界系位置；
- \(\dot c_W\)：质心世界系速度；
- \(f_{ext}\)：除地面反力之外、作用在 CoM 的合外力。

### 6.1 CoM 运动学位置观测

Pinocchio 从 URDF 质量参数和当前关节角得到 base 系 CoM：

\[
c_B=c_B(q).
\]

转换到世界系：

\[
y_c=p_{WB}+R_{WB}c_B(q).
\]

它作为 CoM 位置观测：

\[
z_c=y_c-\hat c_W,
\qquad
H_c=[I\ 0\ 0].
\]

对应代码：

- CoM 运动学：`src/serow/core/src/RobotKinematics.hpp:510-545`
- CoM 位置更新：`src/serow/core/src/CoMEKF.cpp:89-96,240-267`

### 6.2 合并 Ground Reaction Force 与 COP

四足世界系接触力求和：

\[
F_{GRF}=\sum_i f_{W,i}.
\]

当前 Go2 为点足，输入没有足端力矩，所以每只脚的局部 COP 设为零；整体 COP
实际由接触足世界位置按接触概率加权：

\[
p_{COP}=
\frac{\sum_i c_i p_{WF_i}}{\sum_i c_i}.
\]

对应代码：
`src/serow/core/src/Serow.cpp:1143-1175`。

### 6.3 质心动力学

一般的质心平移动力学是：

\[
m\ddot c_W=F_{GRF}+f_{ext}+mg_W.
\]

当前 `CoMEKF` 的水平模型还引入 COP 和质心角动量变化率
\(\dot L=[\dot L_x,\dot L_y,\dot L_z]^\mathsf T\)。记
\(d=c_z-p_{COP,z}\)，代码使用：

\[
\ddot c_x=
\frac{c_x-p_{COP,x}}{md}F_{GRF,z}
+\frac{f_{ext,x}}{m}
-\frac{\dot L_y}{md},
\]

\[
\ddot c_y=
\frac{c_y-p_{COP,y}}{md}F_{GRF,z}
+\frac{f_{ext,y}}{m}
+\frac{\dot L_x}{md},
\]

\[
\ddot c_z=
\frac{F_{GRF,z}+f_{ext,z}}{m}-g.
\]

状态连续模型为：

\[
\dot x_c=
\begin{bmatrix}
\dot c_W\\
\ddot c_W\\
0
\end{bmatrix},
\]

也就是把外力建模为随机游走。

对应代码：
`src/serow/core/src/CoMEKF.cpp:139-180`。

### 6.4 IMU 构造的 CoM 加速度观测

设 \(r=R_{WB}c_B\)，代码由 base 线加速度、角速度和角加速度近似：

\[
y_{\ddot c}
=a_{WB}
+\omega_{WB}\times(\omega_{WB}\times r)
+\alpha_{WB}\times r.
\]

然后将该观测与上一节的质心动力学预测比较，用创新同时修正：

- CoM 位置；
- CoM 速度；
- 外力 \(f_{ext}\)。

对应代码：
`src/serow/core/src/CoMEKF.cpp:98-135,183-237`。

注意当前公式没有显式加入 \(2\omega\times\dot r+\ddot r\) 等完整相对运动项；
这是当前源码采用的近似模型，不应把它描述为完整刚体点加速度公式。

---

## 7. 当前输入最终能得到什么

### 7.1 demo 实际写入输出 MCAP 的量

输出 topic 为 `serow_predictions`，包含：

- base 世界系位置 \(p_{WB}\)；
- base 世界系姿态 \(R_{WB}\)；
- base 世界系线速度 \(v_{WB}\)；
- CoM 世界系位置 \(c_W\)；
- CoM 世界系速度 \(\dot c_W\)；
- CoM 外力 \(f_{ext}\)；
- 加速度计 bias \(b_a\)；
- 陀螺仪 bias \(b_g\)；
- 四足测量接触力（局部足端系）；
- 四足 SEROW 状态中的接触力（世界系和转换回局部足端系）；
- 每只脚世界系姿态。

输出构造代码：
`src/serow/evaluation/mujoco_test/quadruped/go2_mujoco_test.cpp:244-325`。

### 7.2 SEROW 内部还能提供，但当前 demo 未全部写出的量

`State` 还维护：

- base 角速度、线加速度、角加速度；
- base 局部线速度；
- 每只脚世界系位置、姿态、线速度、角速度；
- 接触概率和二值接触状态；
- COP；
- CoM 加速度、角动量和角动量变化率；
- 各状态 covariance。

接口定义：
`src/serow/core/src/State.hpp:34-248`。

### 7.3 “观测量”和“估计状态”不要混用

- 足端相对位姿、足端相对速度：由关节编码器和 URDF直接计算的运动学量；
- 接触概率：由足端竖直力构造的启发式连续量；
- 腿部 base 速度：由“接触脚静止”物理假设构造的伪量测；
- base 位姿、速度、IMU bias：Left-Invariant EKF 状态；
- CoM 位置、速度、外力：CoM EKF 状态；
- `base_ground_truth`：仿真评估真值，不是估计器观测。

---

## 8. 当前系统不能可靠观测什么

### 8.1 绝对世界位置

没有 GPS、视觉、LiDAR 或 motion capture 的绝对位置输入。初始世界原点是估计器
自己定义的，输出是相对轨迹。接触约束能减慢漂移，但不能提供外部全局位置基准。

### 8.2 绝对 yaw

加速度计只能提供重力方向，无法区分绕重力轴的旋转；接触运动学通常也不能提供
绝对航向。没有磁力计或外部 odometry 时，yaw 仍会漂移。

### 8.3 滑动接触

核心模型假设 \(v_{WF_i}\approx0\)。若脚在地面滑动，即使竖直力很大，估计器仍会
把滑动误解释成 base 运动，从而产生位置和速度误差。足端力大只代表承重，不保证
无滑动。

### 8.4 腾空阶段

当 \(\sum_i c_i\) 很小时，没有腿部速度更新，系统接近纯 IMU 积分，速度和位置
不确定性会快速增长。

### 8.5 外力估计的可辨识性

\(f_{ext}\) 是模型残差意义上的外力。接触力误差、质量/惯量误差、base 状态误差、
CoM 模型近似和真实外力都可能进入该状态，所以它不等价于独立力传感器真值。

### 8.6 当前 demo 的“estimated contact force”并非重新估计

当前配置 `estimate_contact_wrench=false`，SEROW 直接接收 MCAP 的
`feet_forces`。输出中的 `estimated_contact_forces_world/local` 是状态中保存并
做坐标变换后的接触力，不应解读成 GMO 从关节 effort 独立反算出的另一套力。

如果切换源码常量 `USE_JOINT_EFFORT_CONTACT_WRENCH=true`，还必须同步将配置改为
`estimate_contact_wrench=true`，才会启用关节 effort + generalized momentum
observer 的另一条链路。

---

## 9. 从一帧数据到输出的完整调用链

```text
/robot_state JSON
  ├─ IMU acceleration + gyro
  │    ├─ 初始静止 bias 标定
  │    ├─ Mahony 姿态
  │    └─ Left-Invariant EKF 惯性预测
  │
  ├─ 12 个 joint position
  │    ├─ 平滑微分得到 joint velocity
  │    └─ Pinocchio 正运动学 / Jacobian
  │         ├─ 足端相对位姿与速度
  │         ├─ CoM 相对位置
  │         └─ CoM 角动量及其变化率
  │
  ├─ 4 个 feet_forces
  │    ├─ 坐标变换
  │    ├─ 中值滤波
  │    ├─ 接触概率
  │    ├─ GRF 求和
  │    └─ COP
  │
  ├─ 接触概率 + 足端运动学 + Mahony 姿态
  │    └─ LegOdometry
  │         ├─ 接触锚点 / IMP
  │         └─ base 线速度伪量测
  │
  ├─ Left-Invariant EKF
  │    ├─ IMU predict
  │    ├─ leg velocity update
  │    └─ terrain height update
  │
  └─ CoMEKF
       ├─ GRF/COP 动力学 predict
       ├─ IMU 构造的 CoM acceleration update
       └─ kinematic CoM position update

输出：base state + CoM state + IMU biases + contact diagnostics
```

主流程的源码顺序：
`src/serow/core/src/Serow.cpp:1347-1409`。

---

## 10. 初学者建议的阅读顺序

1. 先掌握刚体坐标变换：
   \(p_W=p_{WB}+R_{WB}p_B\)。
2. 再推导接触脚速度约束：
   \(0=v+\omega\times Rp+RJ\dot q\)。
3. 阅读 `LegOdometry.cpp:121-188`，看多足接触如何加权。
4. 阅读 `LeftInvariantEKF.cpp:190-221`，理解 IMU 如何预测状态。
5. 阅读 `LeftInvariantEKF.cpp:650-687`，理解腿部速度如何修正预测。
6. 最后阅读 `CoMEKF.cpp`，把 base 里程计与质心动力学分开理解。

建议先把 base 里程计学明白，再学习 CoM 和外力估计。后者不是 base odometry
成立所必需的，而是 SEROW 在基础里程计之上增加的动力学状态估计。

---

## 11. 源码索引

| 主题 | 当前 workspace 代码 |
|---|---|
| MCAP 输入解析 | `src/serow/evaluation/mujoco_test/quadruped/go2_mujoco_test.cpp` |
| Go2 参数 | `src/serow/config/go2.json` |
| Go2 模型 | `src/serow/urdf/go2.urdf` |
| 总流程 | `src/serow/core/src/Serow.cpp` |
| 输入数据结构 | `src/serow/core/src/Measurement.hpp` |
| 输出状态结构 | `src/serow/core/src/State.hpp` |
| Pinocchio 运动学 | `src/serow/core/src/RobotKinematics.hpp` |
| 姿态估计 | `src/serow/core/src/Mahony.hpp` |
| 接触力滤波 | `src/serow/core/src/ContactDetector.hpp` |
| 腿部里程计 | `src/serow/core/src/LegOdometry.cpp` |
| Base Left-Invariant EKF | `src/serow/core/src/LeftInvariantEKF.cpp` |
| CoM 与外力 EKF | `src/serow/core/src/CoMEKF.cpp` |

## 12. 最小结论

当前 Go2 slope demo 的有效传感器输入是：

\[
\boxed{\text{IMU}+\text{12 关节角}+\text{四足三维接触力}+\text{URDF}}
\]

由它们构造的关键观测是：

\[
\boxed{
\text{足端运动学}
\rightarrow
\text{接触概率}
\rightarrow
\text{接触脚静止约束}
\rightarrow
\text{base 速度观测}
}
\]

最终可估计：

\[
\boxed{
\text{base 位姿/速度}
+\text{IMU bias}
+\text{足端/接触状态}
+\text{CoM 状态}
+\text{外力}
}
\]

但在没有外部里程计时，绝对位置和绝对 yaw 不可观，接触滑动与腾空阶段仍会导致
漂移。这是当前传感器组合的物理可观性限制，不是单纯调 EKF 参数就能消除的问题。
