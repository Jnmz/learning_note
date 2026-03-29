# VAE 与 ELBO 笔记

## 元信息

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Auto-Encoding Variational Bayes
  - 潜变量建模相关入门资料

## 一句话总结

VAE 用 amortized variational approximation 去替代潜变量最大似然中的不可解后验，而 ELBO 正是把这种近似变成可训练目标的那个下界，它由重建项和 KL 正则项共同组成。

## 背景 / 问题设定

设我们想通过引入潜变量 \( z \) 来建模数据分布 \( p_{\text{data}}(x) \)。生成模型写成

\[
p_\theta(x,z)=p(z)p_\theta(x\mid z),
\]

其中：

- \( p(z) \) 是一个简单先验，通常取 \( \mathcal{N}(0,I) \)；
- \( p_\theta(x\mid z) \) 是 decoder；
- 边缘似然为

\[
p_\theta(x)=\int p_\theta(x,z)\,dz = \int p(z)p_\theta(x\mid z)\,dz.
\]

核心困难在于，这个积分通常不可解，而精确后验

\[
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}
\]

同样不可解，因为它也依赖归一化量 \( p_\theta(x) \)。

VAE 的做法是引入一个近似后验 \( q_\phi(z\mid x) \)，通常也叫 encoder，然后去优化 \( \log p_\theta(x) \) 的一个下界。

## 记号

- \( x \)：观测数据。
- \( z \)：潜变量。
- \( p(z) \)：潜变量先验。
- \( p_\theta(x\mid z) \)：decoder / 生成条件分布。
- \( p_\theta(x,z)=p(z)p_\theta(x\mid z) \)：联合分布。
- \( p_\theta(z\mid x) \)：真实后验。
- \( q_\phi(z\mid x) \)：变分后验 / encoder。
- \( \theta \)：decoder 参数。
- \( \phi \)：encoder 参数。
- \( \mathcal{L}(x;\theta,\phi) \)：单个样本的 ELBO。
- \( D_{\mathrm{KL}}(q\|p) \)：KL 散度。

## 核心思想

VAE 的故事由三个紧密相连的步骤组成：

1. 把 \( \log p_\theta(x) \) 改写成一个包含 \( q_\phi(z\mid x) \) 的形式；
2. 推导出一个下界，并说明这个下界与真实对数似然之间的差距正好是真实后验上的 KL；
3. 选取高斯形式的 \( q_\phi(z\mid x) \)，再用重参数化让梯度可以穿过潜变量采样过程。

这样一来，潜变量模型就变成了一个可以端到端联合训练的 encoder-decoder 系统。

## 详细推导

### 推导块 1：从对数似然到 ELBO

从边缘对数似然开始：

\[
\log p_\theta(x)=\log \int p_\theta(x,z)\,dz.
\]

引入变分后验 \( q_\phi(z\mid x) \)。因为它是 \( z \) 上的一个密度，可以在积分中乘除同一个量：

\[
\log p_\theta(x)
= \log \int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
\]

把积分看作在 \( q_\phi(z\mid x) \) 下的期望：

\[
\log p_\theta(x)
= \log \mathbb{E}_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\]

现在使用 Jensen 不等式。因为 \( \log \) 是凹函数，

\[
\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y].
\]

于是

\[
\log p_\theta(x)
\ge
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\]

把右边定义为 evidence lower bound：

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right].
\]

这是第一个关键推导：ELBO 不是拍脑袋拼出来的 loss，而是对边缘对数似然直接应用 Jensen 不等式后得到的下界。

### 推导块 2：为什么 ELBO 和对数似然之间的差距是真实后验上的 KL

为了看清 ELBO 真正在优化什么，从 Bayes 公式出发：

\[
p_\theta(z\mid x)=\frac{p_\theta(x,z)}{p_\theta(x)}.
\]

先取对数：

\[
\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x).
\]

整理得到

\[
\log p_\theta(x)=\log p_\theta(x,z)-\log p_\theta(z\mid x).
\]

然后人为地加减 \( \log q_\phi(z\mid x) \)：

\[
\log p_\theta(x)
= \log p_\theta(x,z)-\log q_\phi(z\mid x)
+ \log q_\phi(z\mid x)-\log p_\theta(z\mid x).
\]

对 \( q_\phi(z\mid x) \) 取期望：

\[
\log p_\theta(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)-\log q_\phi(z\mid x)
\right]
\]

\[
\qquad
+ \mathbb{E}_{q_\phi(z\mid x)}
\left[
\log q_\phi(z\mid x)-\log p_\theta(z\mid x)
\right].
\]

识别两个项：

\[
\log p_\theta(x)
= \mathcal{L}(x;\theta,\phi)
+ D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\bigr).
\]

由于 KL 散度总是非负，因此

\[
\mathcal{L}(x;\theta,\phi)\le \log p_\theta(x).
\]

这个恒等式比单纯的 Jensen 推导信息更多：

- 最大化 ELBO 会提升对数似然的下界；
- 同时也在逼迫 \( q_\phi(z\mid x) \) 接近真实后验 \( p_\theta(z\mid x) \)。

### 推导块 3：把 ELBO 拆成重建项与 KL 项

现在使用联合分布分解

\[
p_\theta(x,z)=p(z)p_\theta(x\mid z).
\]

代回 ELBO：

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p(z)+\log p_\theta(x\mid z)-\log q_\phi(z\mid x)
\right].
\]

把重建项和正则项分组：

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+ \mathbb{E}_{q_\phi(z\mid x)}[\log p(z)-\log q_\phi(z\mid x)].
\]

第二项正好就是一个负 KL：

\[
\mathbb{E}_{q_\phi(z\mid x)}[\log p(z)-\log q_\phi(z\mid x)]
= -D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr).
\]

因此标准 VAE 目标为

\[
\mathcal{L}(x;\theta,\phi)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr).
\]

这就是常见的分解：

- **重建项**：\( \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] \)，鼓励 decoder 更好地解释数据。
- **KL 正则项**：\( D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr) \)，鼓励 encoder 的后验接近简单先验。

所以 ELBO 并不只是“重建加正则”这么粗糙，它是从近似最大似然推出来的严格目标。

### 推导块 4：高斯 encoder 的重参数化技巧

如果 \( q_\phi(z\mid x) \) 依赖参数 \( \phi \)，那么直接从

\[
z\sim q_\phi(z\mid x)
\]

采样似乎会阻断关于 \( \phi \) 的反向传播。VAE 用重参数化来解决这个问题。

假设 encoder 输出对角高斯：

\[
q_\phi(z\mid x)
= \mathcal{N}\bigl(z;\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\bigr).
\]

与其直接从该分布采样，不如先采样

\[
\epsilon \sim \mathcal{N}(0,I)
\]

然后定义

\[
z = \mu_\phi(x) + \sigma_\phi(x)\odot \epsilon.
\]

这本质上是高斯采样的一次变量代换：先从标准正态采样，再做平移和缩放。

此时重建期望可以改写为

\[
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
=
\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}
\left[
\log p_\theta\bigl(x\mid \mu_\phi(x)+\sigma_\phi(x)\odot \epsilon\bigr)
\right].
\]

现在随机性被完全隔离在 \( \epsilon \) 中，而 \( \epsilon \) 与 \( \phi \) 无关。因此，关于 \( \phi \) 的梯度可以穿过确定性映射

\[
z(\epsilon,x,\phi)=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon
\]

正常传播。这就是 VAE 能够使用标准随机梯度方法训练的关键。

### 推导块 5：对角高斯后验下的闭式 KL

当

\[
q_\phi(z\mid x)=\mathcal{N}\bigl(z;\mu,\operatorname{diag}(\sigma^2)\bigr),
\qquad
p(z)=\mathcal{N}(0,I),
\]

KL 散度有闭式表达。先写出高斯 KL 的标准公式：

\[
D_{\mathrm{KL}}(\mathcal{N}(\mu_q,\Sigma_q)\,\|\,\mathcal{N}(\mu_p,\Sigma_p))
=
\frac{1}{2}
\left[
\log\frac{|\Sigma_p|}{|\Sigma_q|}
-k
+ \operatorname{tr}(\Sigma_p^{-1}\Sigma_q)
+ (\mu_p-\mu_q)^\top \Sigma_p^{-1}(\mu_p-\mu_q)
\right].
\]

对这里的设定，\( \mu_p=0 \)、\( \Sigma_p=I \)、\( \Sigma_q=\operatorname{diag}(\sigma^2) \)，因此：

- \( |\Sigma_p|=1 \)；
- \( |\Sigma_q|=\prod_{j=1}^k \sigma_j^2 \)；
- \( \Sigma_p^{-1}=I \)；
- \( \operatorname{tr}(\Sigma_q)=\sum_{j=1}^k \sigma_j^2 \)；
- \( \mu^\top \mu=\sum_{j=1}^k \mu_j^2 \)。

把这些代回 KL 公式：

\[
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
= \frac{1}{2}
\left[
-\log |\Sigma_q|
-k
+ \operatorname{tr}(\Sigma_q)
+ \mu^\top \mu
\right].
\]

又因为

\[
\log |\Sigma_q|=\sum_{j=1}^k \log \sigma_j^2,
\]

最终得到

\[
D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
=
\frac{1}{2}\sum_{j=1}^k
\left(
\mu_j^2+\sigma_j^2-\log \sigma_j^2-1
\right).
\]

这就是大多数实现里直接使用的闭式 KL。

## 直觉 / 理解

- Decoder 希望潜变量 \( z \) 足够有信息，从而能把 \( x \) 重建好。
- KL 项则阻止 encoder 把每个样本都映射到潜空间中完全孤立的位置。
- 先验 \( p(z)=\mathcal{N}(0,I) \) 让潜空间足够平滑，从而真正可采样。

我通常把 ELBO 理解成两股力量之间的平衡：

- “让 \( z \) 足够有信息，能够解释 \( x \)”；
- “让编码后的分布足够简单，便于采样与插值”。

如果重建项太强，潜空间会变得很不规则；如果 KL 项太强，模型又可能忽略 \( z \)，出现 posterior collapse。

## 与其他方法的关系

### 与经典变分推断的关系

VAE 可以看成 amortized variational inference：

- 经典 VI 为每个数据点单独优化一个变分分布；
- VAE 学的是一个共享 encoder 网络 \( q_\phi(z\mid x) \)，统一预测所有数据点的变分参数。

所以读 VAE 时，可以把它理解成“带共享推断机制的变分推断”。

### 与 Diffusion Models 的关系

Diffusion models 通常不会像 VAE 那样直接优化一个潜变量 ELBO，但从概念上比较仍然很有帮助：

- 两者都会引入辅助潜变量；
- 两者都会从似然相关的分解或下界出发，构造可训练目标；
- 两者都把困难的密度建模转写成更容易的局部预测问题。

和本仓库里的 diffusion 笔记相比，VAE 更强调潜变量概率建模和后验近似，而 diffusion 更强调去噪路径和反向动力学。

### 与 Flow-Based Models 的关系

Normalizing flows 追求的是通过可逆变换和精确的 change-of-variables 公式来获得精确似然；VAE 则接受近似推断：

- flows：精确似然，但有可逆性约束；
- VAE：后验近似，但 encoder-decoder 设计更灵活。

所以 VAE 用精确性换取了灵活性和更自然的潜变量语义。

## 我的笔记 / 开放问题

- 恒等式 \( \log p_\theta(x)=\mathcal{L}(x;\theta,\phi)+D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)) \) 是整个方法的概念中心。只要这一步想清楚，VAE 就会神秘感大减。
- 重参数化不只是实现细节，它是“随机潜变量也能做梯度训练”这件事真正成立的关键。
- 一个很适合后续补充的方向，是从 ELBO 权衡的角度比较 VAE、beta-VAE 和 hierarchical VAE。

## 参考资料

- [Kingma and Welling (2014), *Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114)
- [Doersch (2016), *Tutorial on Variational Autoencoders*](https://arxiv.org/abs/1606.05908)
