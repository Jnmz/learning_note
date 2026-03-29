# Flow Matching 笔记

## 元信息

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Flow Matching for Generative Modeling
  - 用于比较的 diffusion 与 score-based 教程

## 一句话总结

Flow matching 学习一个随时间变化的速度场，让 ODE 把简单基分布运输到数据分布，而它的训练目标之所以成立，是因为条件速度回归在最优时会恢复出边缘 transport field。

## 背景 / 问题设定

DDPM 和 score-based model 往往从随机加噪过程出发，而 flow matching 更直接地从 transport 视角出发：

1. 先选一条连接简单源分布与数据分布的概率路径 \( \{p_t\}_{t\in[0,1]} \)；
2. 再用速度场描述这条路径；
3. 最后直接回归这个速度场。

这种看法的吸引力在于，它把生成过程更明确地写成连续时间的概率质量运输，而不是反向去噪。

## 记号

- \( p_0 \)：源分布，常取 \( \mathcal{N}(0,I) \)。
- \( p_1 \)：目标数据分布。
- \( p_t \)：时刻 \( t\in[0,1] \) 的中间边缘分布。
- \( x_t \)：粒子在时刻 \( t \) 的状态。
- \( v_t(x) \)：边缘速度场。
- \( p_t(x\mid x_0,x_1) \)：条件概率路径。
- \( u_t(x\mid x_0,x_1) \)：条件速度场。
- \( \mu_t(x_0,x_1) \)：条件路径均值。
- \( \sigma_t \)：条件路径尺度。
- \( \pi(dx_0,dx_1) \)：端点耦合分布。

## 核心思想

Flow matching 从下面的 ODE 出发：

\[
\frac{dx_t}{dt}=v_t(x_t).
\]

如果粒子按照这条 ODE 运动，且初值满足 \( x_0\sim p_0 \)，那么诱导出的密度就会从 \( p_0 \) 演化到 \( p_1 \)。真正困难的地方在于怎样学习 \( v_t \)。Flow matching 的做法是，先构造一条条件路径，使得目标速度在解析上可得。

## 详细推导

### 推导块 1：从 ODE 动力学到 continuity equation

假设粒子满足

\[
\dot{x}_t=v_t(x_t).
\]

设 \( p_t(x) \) 是 \( x_t \) 的密度。对任意光滑测试函数 \( \varphi(x) \)，有

\[
\frac{d}{dt}\mathbb{E}_{p_t}[\varphi(x)]
= \frac{d}{dt}\int \varphi(x)p_t(x)\,dx.
\]

沿着粒子轨迹使用链式法则：

\[
\frac{d}{dt}\varphi(x_t)
= \nabla \varphi(x_t)^\top \dot{x}_t
= \nabla \varphi(x_t)^\top v_t(x_t).
\]

对其取期望：

\[
\frac{d}{dt}\int \varphi(x)p_t(x)\,dx
= \int \nabla \varphi(x)^\top v_t(x)p_t(x)\,dx.
\]

再做分部积分：

\[
\int \nabla \varphi(x)^\top v_t(x)p_t(x)\,dx
= -\int \varphi(x)\,\nabla\cdot\bigl(p_t(x)v_t(x)\bigr)\,dx,
\]

并假设边界项消失。

于是

\[
\int \varphi(x)\,\partial_t p_t(x)\,dx
= -\int \varphi(x)\,\nabla\cdot\bigl(p_t(x)v_t(x)\bigr)\,dx.
\]

因为这对任意测试函数 \( \varphi \) 都成立，所以得到 continuity equation：

\[
\partial_t p_t(x)+\nabla\cdot\bigl(p_t(x)v_t(x)\bigr)=0.
\]

这就是 flow matching 背后的密度守恒定律。

### 推导块 2：条件高斯路径与条件速度

选择一条连接端点 \( x_0 \sim p_0 \) 与 \( x_1 \sim p_1 \) 的条件路径：

\[
p_t(x\mid x_0,x_1)
= \mathcal{N}\bigl(x;\mu_t(x_0,x_1),\sigma_t^2 I\bigr).
\]

对它做重参数化：

\[
x_t = \mu_t(x_0,x_1) + \sigma_t \epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

对时间求导：

\[
\frac{dx_t}{dt}
= \partial_t \mu_t(x_0,x_1)+\dot{\sigma}_t\epsilon.
\]

再由重参数化公式解出 \( \epsilon \)：

\[
\epsilon = \frac{x_t-\mu_t(x_0,x_1)}{\sigma_t}.
\]

代回即可得到条件速度目标：

\[
u_t(x_t\mid x_0,x_1)
= \partial_t \mu_t(x_0,x_1)
+ \frac{\dot{\sigma}_t}{\sigma_t}\bigl(x_t-\mu_t(x_0,x_1)\bigr).
\]

这个目标能写成闭式，正是因为我们事先把条件路径选成了解析友好的形式。

对于常见的线性均值路径

\[
\mu_t(x_0,x_1)=(1-t)x_0 + tx_1,
\]

有

\[
\partial_t \mu_t(x_0,x_1)=x_1-x_0,
\]

因此

\[
u_t(x_t\mid x_0,x_1)
= (x_1-x_0)
+ \frac{\dot{\sigma}_t}{\sigma_t}\bigl(x_t-\mu_t(x_0,x_1)\bigr).
\]

这里的推导显式使用了重参数化与变量代换。

### 推导块 3：为什么条件速度回归能恢复边缘速度场

训练目标写成

\[
\mathcal{L}_{\mathrm{FM}}
= \mathbb{E}\left[
\|v_\theta(x_t,t)-u_t(x_t\mid x_0,x_1)\|^2
\right],
\]

其中 \( x_t\sim p_t(\cdot\mid x_0,x_1) \)。

固定时间 \( t \) 与位置 \( x \)，点态回归问题为

\[
\min_v \mathbb{E}\left[
\|v-u_t(x_t\mid x_0,x_1)\|^2 \mid x_t=x
\right].
\]

把平方项展开：

\[
\mathbb{E}\left[
\|v-u\|^2 \mid x_t=x
\right]
= \|v\|^2 - 2v^\top \mathbb{E}[u\mid x_t=x] + \mathbb{E}[\|u\|^2\mid x_t=x].
\]

对 \( v \) 求导并令其为零：

\[
2v - 2\mathbb{E}[u\mid x_t=x] = 0.
\]

因此最优预测器为

\[
v_t^\star(x)
= \mathbb{E}\bigl[u_t(x_t\mid x_0,x_1)\mid x_t=x\bigr].
\]

现在检查这个场是否真的推动边缘路径。边缘密度是

\[
p_t(x)=\int p_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1).
\]

对时间求导，并利用条件 continuity equation：

\[
\partial_t p_t(x)
= \int \partial_t p_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1)
\]

\[
= -\int \nabla\cdot\bigl(p_t(x\mid x_0,x_1)u_t(x\mid x_0,x_1)\bigr)\,\pi(dx_0,dx_1).
\]

把 divergence 移到积分外：

\[
\partial_t p_t(x)
= -\nabla\cdot\left(
\int p_t(x\mid x_0,x_1)u_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1)
\right).
\]

根据条件期望的定义，

\[
\int p_t(x\mid x_0,x_1)u_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1)
= p_t(x)\,v_t^\star(x).
\]

因此

\[
\partial_t p_t(x)+\nabla\cdot\bigl(p_t(x)v_t^\star(x)\bigr)=0.
\]

也就是说，回归最优解恰好就是实现整个概率路径的边缘速度场。

## 直觉 / 理解

- continuity equation 表示粒子运动时概率质量守恒。
- 条件路径为我们提供了容易获得的局部监督信号。
- 条件期望则把这些局部信号拼接成了正确的边缘 transport field。

我会在理解 flow matching 时主动停止用“去噪”来想，而转成“概率质量如何移动”的视角，这样会清楚很多。

## 与其他方法的关系

### 与 DDPM 的关系

DDPM 用的是离散随机链，并学习反向去噪转移；flow matching 用的是连续时间概率路径，并学习速度场。两者都属于 path-based generative methods，但局部监督信号不同：

- DDPM 的目标：噪声或后验均值。
- Flow matching 的目标：速度。

可以参见 [DDPM 笔记](./ddpm-notes.md) 来看离散高斯链的版本。

### 与 Score Matching 的关系

Score-based model 学习的是

\[
s_t(x)=\nabla_x\log p_t(x).
\]

对一个 SDE

\[
dx = f(x,t)\,dt + g(t)\,dW_t,
\]

它对应的 probability-flow ODE 为

\[
\frac{dx}{dt}
= f(x,t) - \frac{1}{2}g(t)^2\nabla_x\log p_t(x).
\]

所以 score 场本身也会诱导出一个确定性的速度场。Flow matching 只是直接从这个 ODE 风格的视角出发，而不是先写 SDE 再推出来。这就是它与 [Score Matching 笔记](./score-matching-notes.md) 的主要桥梁。

### ODE、SDE 与 continuity equation

- ODE：描述确定性的粒子运动。
- SDE：描述带扩散噪声的随机粒子运动。
- continuity equation：描述所选动力学下密度如何演化。

它们分别对应轨迹层面、随机层面和密度层面的三种互补描述。

## 我的笔记 / 开放问题

- 条件期望那一步的证明是 flow matching 的概念中心；如果不把这一步看清楚，方法就很像一个任意的回归技巧。
- 当把它和 score-based probability-flow ODE 对照起来时，flow matching 的 transport 视角会特别清楚。
- 一个值得继续补充的方向，是比较 rectified flow、optimal transport flow matching 和经典 score-based probability-flow ODE。

## 参考资料

- [Lipman et al. (2023), *Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/)
