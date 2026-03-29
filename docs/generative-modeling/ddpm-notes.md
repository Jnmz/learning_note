# DDPM 笔记

## 元信息

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Denoising Diffusion Probabilistic Models
  - Lilian Weng 的 diffusion 总览
  - 用户提供的知乎笔记

## 一句话总结

DDPM 定义了一条离散的高斯加噪链并学习它的反向过程，而常见的 epsilon-prediction loss，本质上是由高斯后验匹配出发、经过重参数化后得到的变分目标。

## 背景 / 问题设定

我们想建模样本 \( x_0 \sim q(x_0) \)。DDPM 并不试图一步直接学习 \( q(x_0) \)，而是引入一条潜变量链

\[
x_0 \to x_1 \to \cdots \to x_T
\]

其中前向过程逐步把数据腐蚀成噪声，反向过程再学习如何把噪声还原回数据。

整体策略可以概括为：

1. 先选一条简单的前向高斯 Markov 链；
2. 再写一个带可学习参数的反向高斯链；
3. 用变分下界进行优化；
4. 最后把可训练部分重写成噪声预测问题。

## 记号

- \( x_0 \)：干净数据样本。
- \( x_t \)：扩散步骤 \( t \in \{1,\dots,T\} \) 上的潜变量。
- \( q(x_t\mid x_{t-1}) \)：前向加噪核。
- \( p_\theta(x_{t-1}\mid x_t) \)：学习得到的反向核。
- \( \beta_t \in (0,1) \)：方差日程。
- \( \alpha_t = 1-\beta_t \)。
- \( \bar{\alpha}_t = \prod_{s=1}^t \alpha_s \)。
- \( \epsilon,\epsilon_t \sim \mathcal{N}(0,I) \)：高斯噪声。
- \( \mu_\theta(x_t,t) \)：反向均值。
- \( \Sigma_\theta(x_t,t) \)：反向协方差。

## 核心思想

前向链被设定为

\[
q(x_t\mid x_{t-1})
= \mathcal{N}\bigl(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\bigr).
\]

每一步都保留上一状态的缩放版本，同时注入高斯噪声。经过很多步以后，\( x_T \) 会接近标准高斯。反向模型再使用

\[
p_\theta(x_{t-1}\mid x_t)
= \mathcal{N}\bigl(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\bigr)
\]

把噪声逐步还原成数据。

核心推导路线是：

- 先推出直接的前向边缘分布 \( q(x_t\mid x_0) \)；
- 再推出前向后验 \( q(x_{t-1}\mid x_t,x_0) \)；
- 最后说明为什么 ELBO 会化成 epsilon prediction。

## 详细推导

### 推导块 1：\( q(x_t \mid x_0) \) 的闭式表达

从前向重参数化开始：

\[
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal{N}(0,I).
\]

展开一步：

\[
\begin{aligned}
x_t
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-1}\right)
+ \sqrt{1-\alpha_t}\epsilon_t \\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
+ \sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}
+ \sqrt{1-\alpha_t}\epsilon_t.
\end{aligned}
\]

继续递归展开可得

\[
x_t
= \sqrt{\bar{\alpha}_t}x_0
+ \sum_{s=1}^t
\left(
\sqrt{1-\alpha_s}\prod_{r=s+1}^t \sqrt{\alpha_r}
\right)\epsilon_s.
\]

因为这是若干独立高斯变量的线性组合，所以在给定 \( x_0 \) 时它仍然是高斯分布。其均值为

\[
\mathbb{E}[x_t\mid x_0]=\sqrt{\bar{\alpha}_t}x_0.
\]

其协方差为

\[
\operatorname{Var}(x_t\mid x_0)
= \sum_{s=1}^t
\left(
(1-\alpha_s)\prod_{r=s+1}^t \alpha_r
\right)I.
\]

现在使用 telescoping identity

\[
\sum_{s=1}^t
\left(
(1-\alpha_s)\prod_{r=s+1}^t \alpha_r
\right)
= 1-\prod_{s=1}^t \alpha_s
= 1-\bar{\alpha}_t.
\]

因此

\[
q(x_t\mid x_0)
= \mathcal{N}\bigl(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\bigr).
\]

等价地，根据高斯重参数化，

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

这个公式非常关键，因为它允许我们从 \( x_0 \) 一步直接采样 \( x_t \)。

### 推导块 2：后验 \( q(x_{t-1}\mid x_t,x_0) \)

现在来推导前向链的精确反向条件分布。利用 Bayes 公式和 Markov 性：

\[
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0).
\]

两个高斯因子分别是

\[
q(x_t\mid x_{t-1})
\propto
\exp\left(
-\frac{1}{2(1-\alpha_t)}\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2
\right),
\]

\[
q(x_{t-1}\mid x_0)
\propto
\exp\left(
-\frac{1}{2(1-\bar{\alpha}_{t-1})}
\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
\right).
\]

把关于 \( x_{t-1} \) 的二次项展开：

\[
\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2
= \|x_t\|^2 - 2\sqrt{\alpha_t}x_t^\top x_{t-1} + \alpha_t\|x_{t-1}\|^2,
\]

\[
\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
= \|x_{t-1}\|^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0^\top x_{t-1}
+ \bar{\alpha}_{t-1}\|x_0\|^2.
\]

只保留依赖 \( x_{t-1} \) 的项，得到对数密度

\[
\log q(x_{t-1}\mid x_t,x_0)
=
-\frac{1}{2}
\left[
\frac{\alpha_t}{1-\alpha_t}
+ \frac{1}{1-\bar{\alpha}_{t-1}}
\right]\|x_{t-1}\|^2
\]

\[
\qquad
+ \left[
\frac{\sqrt{\alpha_t}}{1-\alpha_t}x_t
+ \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
\right]^\top x_{t-1}
+ C.
\]

再利用 \( \bar{\alpha}_t=\alpha_t\bar{\alpha}_{t-1} \) 与 \( 1-\alpha_t=\beta_t \) 来化简二次项系数：

\[
\frac{\alpha_t}{1-\alpha_t}
+ \frac{1}{1-\bar{\alpha}_{t-1}}
= \frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}.
\]

配方后可得

\[
q(x_{t-1}\mid x_t,x_0)
= \mathcal{N}\bigl(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t I\bigr),
\]

其中

\[
\tilde{\beta}_t
= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
\]

\[
\tilde{\mu}_t(x_t,x_0)
= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t.
\]

所以反向模型真正需要逼近的，其实就是一个已知高斯后验的均值和方差。

### 推导块 3：从 ELBO 到 epsilon prediction

反向生成模型写成

\[
p_\theta(x_{0:T})
= p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),
\qquad p(x_T)=\mathcal{N}(0,I).
\]

把 \( q(x_{1:T}\mid x_0) \) 当作变分分布，并使用 Jensen 不等式：

\[
\log p_\theta(x_0)
= \log \int q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}dx_{1:T}
\ge
\mathbb{E}_q\left[
\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
\]

把分子分母展开：

\[
\mathcal{L}_{\mathrm{vlb}}
= \mathbb{E}_q\left[
\log p(x_T)
+ \sum_{t=1}^T \log p_\theta(x_{t-1}\mid x_t)
- \sum_{t=1}^T \log q(x_t\mid x_{t-1})
\right].
\]

整理成条件分布形式：

\[
\mathcal{L}_{\mathrm{vlb}}
= \mathbb{E}_q\Bigl[
D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))
+ \sum_{t=2}^T D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)
- \log p_\theta(x_0\mid x_1)
\Bigr].
\]

如果反向方差固定，那么每一项 KL 都会化成加权的均方误差：

\[
\mathcal{L}_t
\propto
\mathbb{E}\left[
\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)\|^2
\right].
\]

现在利用直接前向边缘分布

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
\]

解出 \( x_0 \)：

\[
x_0
= \frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
\right).
\]

把它代入 \( \tilde{\mu}_t(x_t,x_0) \)：

\[
\begin{aligned}
\tilde{\mu}_t(x_t,x_0)
&=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}
\cdot
\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
\right)
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \\
&=
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}x_t
- \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\epsilon
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t.
\end{aligned}
\]

合并 \( x_t \) 的系数：

\[
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
= \frac{1}{\sqrt{\alpha_t}}.
\]

因此

\[
\tilde{\mu}_t(x_t,x_0)
= \frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon
\right).
\]

如果网络预测 \( \epsilon_\theta(x_t,t) \)，定义

\[
\mu_\theta(x_t,t)
= \frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right).
\]

那么可训练目标就会变成一个只差时间权重的噪声回归：

\[
\mathcal{L}_{\mathrm{simple}}
= \mathbb{E}_{t,x_0,\epsilon}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right].
\]

这就是为什么 DDPM 通常被实现成噪声预测模型，尽管它最初是从一个 ELBO 出发推出来的。

## 直觉 / 理解

- 前向链把无标签数据制造成大量监督式去噪任务。
- 反向后验告诉我们，如果知道 \( x_0 \)，理想的去噪器应该做什么。
- 一旦 \( x_t \) 和 \( t \) 已知，\( x_0 \) 与 \( \epsilon \) 其实是可以互相换写的，所以 epsilon-prediction 就成立。

我觉得理解 DDPM 最舒服的方式，是把它看成“一个经过精心设计的高斯潜变量模型”。

## 与其他方法的关系

### 与 Score Matching 的关系

由于

\[
x_t\mid x_0 \sim \mathcal{N}\bigl(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\bigr),
\]

条件核的 score 为

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}.
\]

再利用 \( x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\epsilon \)，可得

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
\]

因此预测 \( \epsilon \) 等价于预测一个缩放后的 score。这就是它与 [Score Matching 笔记](./score-matching-notes.md) 之间最精确的桥梁。

### 与 Flow Matching 的关系

DDPM 是离散的随机生成模型；flow matching 是连续的 transport 模型。但两者在高层结构上非常相似：

- 先选一条从数据到噪声的路径；
- 再在路径上定义局部监督信号；
- 最后在采样时积分学习到的反向动力学。

DDPM 中的局部信号是噪声或后验均值，flow matching 中的局部信号是速度。可以继续看 [Flow Matching 笔记](./flow-matching-notes.md)。

## 我的笔记 / 开放问题

- DDPM 最漂亮的部分不是网络结构，而是前向后验里的那套高斯代数。
- ELBO 解释了原始方法的来龙去脉，但很多后续改进更关心更好的参数化方式，而不执着于原始下界本身。
- 一个自然的后续笔记方向，是把 DDPM、DDIM 和 probability-flow ODE sampling 放在同一页里对比。

## 参考资料

- [Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Zhihu note on diffusion models](https://zhuanlan.zhihu.com/p/689677093)
