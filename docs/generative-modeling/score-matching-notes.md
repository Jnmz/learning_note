# Score Matching 笔记

## 元信息

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Score matching
  - score-based generative modeling 相关教程
  - 连接 epsilon prediction 与 score 的 diffusion 教程

## 一句话总结

Score matching 学习的是对数密度的梯度而不是密度本身，因此训练目标里会自然消掉 partition function，并进一步通向 denoising score matching、diffusion training 和 score-based sampling。

## 背景 / 问题设定

设模型是一个 energy-based model：

\[
p_\theta(x)=\frac{1}{Z_\theta}\exp(-E_\theta(x)).
\]

最大似然通常很难做，因为 \( Z_\theta \) 往往不可积或难以计算。Score matching 的做法是，不直接学习密度本身，而是学习它关于 \( x \) 的梯度。

这个视角很重要，因为它是从经典能量模型通向现代 diffusion 与 score-based generative modeling 的一条非常干净的概念路径。

## 记号

- \( p_{\text{data}}(x) \)：数据分布。
- \( p_\theta(x) \)：模型分布。
- \( E_\theta(x) \)：能量函数。
- \( Z_\theta \)：配分函数。
- \( s_\theta(x)=\nabla_x\log p_\theta(x) \)：模型 score。
- \( s_{\text{data}}(x)=\nabla_x\log p_{\text{data}}(x) \)：数据 score。
- \( \nabla\cdot s_\theta(x) \)：score 场的散度。
- \( \sigma \)：高斯扰动尺度。
- \( \tilde{x} \)：加噪观测。

## 核心思想

一个密度 \( p(x) \) 的 score 定义为

\[
s_p(x)=\nabla_x\log p(x).
\]

它指向对数密度增长最快的方向。如果我们在整个空间里都知道这个向量场，那么即便归一化密度难以显式计算，我们仍然掌握了分布的大量几何信息。

这里最核心的数学故事是：

1. 对 \( \log p_\theta(x) \) 求导会消去 partition function；
2. Fisher divergence 可以改写成不显式依赖数据 score 的形式；
3. 用高斯噪声平滑数据之后，会得到 denoising score matching；
4. diffusion 训练本质上就是带时间索引的 denoising score matching。

## 详细推导

### 推导块 1：score 的定义如何消去 partition function

从

\[
p_\theta(x)=\frac{1}{Z_\theta}\exp(-E_\theta(x))
\]

开始。先取对数：

\[
\log p_\theta(x)=-E_\theta(x)-\log Z_\theta.
\]

再对 \( x \) 求梯度：

\[
\nabla_x\log p_\theta(x)
= \nabla_x\bigl(-E_\theta(x)-\log Z_\theta\bigr).
\]

因为 \( Z_\theta \) 依赖参数，但不依赖自变量 \( x \)，所以

\[
\nabla_x \log Z_\theta = 0.
\]

因此

\[
s_\theta(x)=\nabla_x\log p_\theta(x)=-\nabla_x E_\theta(x).
\]

这就是 score matching 吸引人的第一层原因：它只看局部几何，而不要求显式计算模型的归一化常数。

### 推导块 2：从 Fisher divergence 到 Hyvarinen 目标

理想目标是 Fisher divergence：

\[
J(\theta)
= \frac{1}{2}\int p_{\text{data}}(x)\|s_\theta(x)-s_{\text{data}}(x)\|^2 dx.
\]

把平方展开：

\[
\begin{aligned}
J(\theta)
&= \frac{1}{2}\int p_{\text{data}}(x)\|s_\theta(x)\|^2 dx \\
&\quad - \int p_{\text{data}}(x)s_\theta(x)^\top s_{\text{data}}(x)\,dx \\
&\quad + \frac{1}{2}\int p_{\text{data}}(x)\|s_{\text{data}}(x)\|^2 dx.
\end{aligned}
\]

最后一项与 \( \theta \) 无关，所以真正麻烦的是

\[
\int p_{\text{data}}(x)s_\theta(x)^\top s_{\text{data}}(x)\,dx.
\]

现在使用 score 的定义：

\[
s_{\text{data}}(x)=\nabla_x\log p_{\text{data}}(x)
= \frac{\nabla_x p_{\text{data}}(x)}{p_{\text{data}}(x)}.
\]

代入后得到

\[
\int p_{\text{data}}(x)s_\theta(x)^\top s_{\text{data}}(x)\,dx
= \int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\,dx.
\]

接着对每个坐标应用分部积分。对于第 \( i \) 个坐标，

\[
\int s_{\theta,i}(x)\,\partial_i p_{\text{data}}(x)\,dx
= \left[s_{\theta,i}(x)p_{\text{data}}(x)\right]_{\partial\Omega}
- \int p_{\text{data}}(x)\,\partial_i s_{\theta,i}(x)\,dx.
\]

假设边界项为零，则

\[
\int s_{\theta,i}(x)\,\partial_i p_{\text{data}}(x)\,dx
= - \int p_{\text{data}}(x)\,\partial_i s_{\theta,i}(x)\,dx.
\]

对所有坐标求和：

\[
\int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\,dx
= -\int p_{\text{data}}(x)\,\nabla\cdot s_\theta(x)\,dx.
\]

因此与优化相关的目标变成

\[
J(\theta)
= \mathbb{E}_{p_{\text{data}}}
\left[
\frac{1}{2}\|s_\theta(x)\|^2 + \nabla\cdot s_\theta(x)
\right]
+ C.
\]

这就是经典的 score matching 目标。未知的数据 score 已经消失了。

### 推导块 3：带高斯扰动的 denoising score matching

直接做 score matching 在高维时往往并不方便，因为散度项难算，干净数据分布上的 score 也可能不够平滑。所以一个常见做法是先用高斯噪声把数据平滑掉：

\[
\tilde{x}=x+\sigma\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

条件分布是

\[
q_\sigma(\tilde{x}\mid x)
= \mathcal{N}(\tilde{x};x,\sigma^2I).
\]

取对数：

\[
\log q_\sigma(\tilde{x}\mid x)
= -\frac{d}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}\|\tilde{x}-x\|^2.
\]

对 \( \tilde{x} \) 求梯度：

\[
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)
= -\frac{1}{2\sigma^2}\nabla_{\tilde{x}}\|\tilde{x}-x\|^2
= -\frac{\tilde{x}-x}{\sigma^2}.
\]

于是 denoising score matching 的目标可以直接写成

\[
\mathcal{L}_{\mathrm{DSM}}
= \mathbb{E}_{x,\tilde{x}}
\left[
\left\|
s_\theta(\tilde{x},\sigma)
- \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)
\right\|^2
\right]
\]

\[
= \mathbb{E}_{x,\tilde{x}}
\left[
\left\|
s_\theta(\tilde{x},\sigma)
+ \frac{\tilde{x}-x}{\sigma^2}
\right\|^2
\right].
\]

这一步让现代生成模型中的 score estimation 变得可操作。

## 直觉 / 理解

- score 是一个向量场，告诉我们密度在哪个方向上升。
- 经典 score matching 试图直接匹配干净数据上的这个向量场。
- denoising score matching 先把数据模糊化，于是这个向量场会变得更容易学习。

我更喜欢把 denoising score matching 理解成“在被模糊过的数据流形上学习几何”。

## 与其他方法的关系

### 与 DDPM 的关系

扩散模型里常用的扰动形式是

\[
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.
\]

对应的条件 score 为

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}.
\]

由于

\[
x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\epsilon,
\]

我们有

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
\]

因此 DDPM 的 epsilon-prediction 其实就是经过确定性缩放后的 score estimation：

\[
s_\theta(x_t,t)
= -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
\]

这就是它与 [DDPM 笔记](./ddpm-notes.md) 的直接联系。

### 与 Langevin Sampling 的关系

一旦有了 score estimator，就可以通过 Langevin dynamics 采样：

\[
x_{k+1}=x_k+\eta s_\theta(x_k)+\sqrt{2\eta}\,z_k,
\qquad z_k\sim\mathcal{N}(0,I).
\]

梯度项把样本推向高密度区域，噪声项则帮助探索空间。这就是经典的 score-based sampling 思路。

### 与 Flow Matching 的关系

Flow matching 学习的是速度场而不是 score 场，但 score-based diffusion 模型同样会诱导出一个 probability-flow ODE。也就是说，score 场也能定义一个确定性的 transport dynamics。这正是它与 [Flow Matching 笔记](./flow-matching-notes.md) 的主要概念桥梁。

## 我的笔记 / 开放问题

- 真正重要的概念跳跃，不只是 Fisher divergence 本身，而是去噪化之后的改写。
- Score matching 是一种非常“先看局部几何，再谈全局生成”的思路。
- 一个值得后续补充的方向，是把 Langevin、reverse SDE 和 probability-flow ODE sampling 并排放在一起比较。

## 参考资料

- [Aapo Hyvarinen (2005), *Estimation of Non-Normalized Statistical Models by Score Matching*](https://jmlr.org/papers/v6/hyvarinen05a.html)
- [Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Zhihu note on score-based and diffusion models](https://zhuanlan.zhihu.com/p/12591930520)
