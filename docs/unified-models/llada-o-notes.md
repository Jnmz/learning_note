# LLaDA-o 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model
  - LLaDA-o official repository

## 一句话总结

LLaDA-o 的核心不只是“又一个 diffusion 统一模型”，而是把统一对象明确拆成离散 masked diffusion 和连续 diffusion 两种子机制，再通过共享高效 attention backbone 和数据驱动的长度自适应策略，把它们组织成一个更可用的 omni diffusion model。

## 背景 / 问题设定

如果接受“统一模型也可以围绕 diffusion 建立”，接下来马上会遇到两个很实际的问题：

1. 文本理解和视觉生成显然不适合同一种 diffusion 形式。
2. 多模态输出长度差异极大，若没有长度适配机制，统一推理会非常低效。

MMaDA 已经提出一个更宏观的范式问题：diffusion 能否成为 unified foundation model 的中心。但 LLaDA-o 更进一步，它在问：

- 就算 diffusion 可以做统一，内部是否仍需要进一步结构拆分？
- 文本和视觉是否应该共享同一种 diffusion 子机制？
- 如果统一模型面对的任务长度分布非常不同，推理长度和计算预算是否也应该随任务而变？

因此 LLaDA-o 不是简单重复 MMaDA，而是在给 omni-diffusion 方法增加更细的内部结构和更现实的效率机制。它真正瞄准的，是：

\[
\text{how to make omni-diffusion not just possible, but practical}.
\]

## 记号

设：

- 文本序列为 \(x\)
- 图像连续变量或 latent 为 \(y\)
- 共享注意力主干为 \(f_\theta\)
- 离散 masked diffusion 目标为 \(\mathcal{L}_{\text{disc}}\)
- 连续 diffusion 目标为 \(\mathcal{L}_{\text{cont}}\)
- 长度适配相关目标为 \(\mathcal{L}_{\text{len}}\)
- 解码长度或推理预算控制变量为 \(L\)

LLaDA-o 的核心结构可以抽象写成：

\[
\text{text / reasoning tasks}
\rightarrow
\text{discrete diffusion branch},
\]

\[
\text{visual generation tasks}
\rightarrow
\text{continuous diffusion branch},
\]

并通过共享 backbone 汇合为

\[
h = f_\theta(\text{mixed multimodal context}).
\]

同时，长度适配机制可抽象理解为一个任务条件下的预算映射：

\[
L^\star = g(\text{task}, \text{input context}).
\]

## 核心思想

### 1. Mixture of Diffusion

LLaDA-o 不要求所有模态走完全相同的 diffusion 过程，而是把统一建立在一个 Mixture of Diffusion（MoD）框架上：

- 文本理解更适合离散 masked diffusion；
- 视觉生成更适合连续 diffusion。

因此它的统一不是“所有模态同一动力学”，而是“不同 diffusion 子机制在同一个 backbone 中协作”。

### 2. 共享主干而非共享所有细节

不同 diffusion 分支通过共享高效 attention backbone 耦合，因此统一仍然发生在上下文交互层，而不是变量形式层。

这意味着 LLaDA-o 的统一观更接近：

\[
\text{shared contextual computation}
\neq
\text{shared diffusion submechanism}.
\]

### 3. Length Adaptation 是核心工程机制

LLaDA-o 很重要的一点是，它把长度自适配从工程细节提升为方法设计的一部分。这说明作者意识到 unified model 的实际难点常常不是“能不能统一”，而是：

\[
\text{unified after training, can it decode efficiently enough to be useful?}
\]

## 一个简单示意图

```text
text / reasoning input ----------> discrete diffusion branch ----+
                                                                 |
image / visual target ----------> continuous diffusion branch ---+--> shared attention backbone --> outputs
                                                                 |
                                                                 +--> length-adaptive decoding control
```

## 详细推导

### 推导 1：LLaDA-o 的统一不是单一 diffusion，而是 diffusion family 内部的结构化分解

如果一个 omni-diffusion 模型试图让所有模态都服从完全相同的 diffusion 形式，那么很快会遇到问题：

- 文本更接近离散符号对象；
- 图像更接近连续状态对象。

因此 LLaDA-o 不再要求

\[
\text{one diffusion form fits all},
\]

而是改成

\[
\text{one shared backbone} + \text{multiple diffusion branches}.
\]

具体地，对文本 / reasoning 分支，可用离散 masked diffusion 视角描述。设文本序列为 \(x\)，mask 后状态为 \(\tilde{x}\)，则目标是恢复被 mask 的位置：

\[
\mathcal{L}_{\text{disc}}
=
- \mathbb{E}_{M}
\left[
\sum_{i \in M}
\log p_\theta(x_i \mid \tilde{x}, c)
\right].
\]

这里 \(M\) 是 mask 集合，\(c\) 是任务条件与共享上下文。

对视觉分支，则保留连续 diffusion 目标。设真实视觉状态为 \(y_1\)、噪声为 \(y_0\)、加噪状态为 \(y_t\)，则典型连续目标可以写成

\[
\mathcal{L}_{\text{cont}}
=
\mathbb{E}_{y_1, y_0, t}
\left[
\left\|
\epsilon_\theta(y_t, t, c) - y_0
\right\|_2^2
\right].
\]

因此 LLaDA-o 的统一，本质上不是单一公式统一，而是 diffusion family 内部的结构化分解。

### 推导 2：联合目标体现“分支异构，主干共享”

LLaDA-o 的训练可以抽象写成

\[
\mathcal{L}
=
\lambda_{\text{disc}} \mathcal{L}_{\text{disc}}
+
\lambda_{\text{cont}} \mathcal{L}_{\text{cont}}
+
\lambda_{\text{len}} \mathcal{L}_{\text{len}}.
\]

这个式子真正重要的，不是“loss 加权求和”本身，而是参数依赖结构：

\[
\mathcal{L}_{\text{disc}} = \mathcal{L}_{\text{disc}}(f_\theta, \text{disc branch}),
\]

\[
\mathcal{L}_{\text{cont}} = \mathcal{L}_{\text{cont}}(f_\theta, \text{cont branch}),
\]

\[
\mathcal{L}_{\text{len}} = \mathcal{L}_{\text{len}}(f_\theta, g).
\]

这里共享的是 attention backbone \(f_\theta\)，而不是所有 diffusion 细节都共享。这说明 LLaDA-o 对 unified model 的定义更加细化：

\[
\text{unified} = \text{shared contextual backbone with heterogeneous diffusion branches}.
\]

这和 Emu3 那种“统一到单一 NTP 目标”的路线形成鲜明对比，也和 MMaDA 那种更宏观的“统一 diffusion 范式”相比，多了一层内部机制分工。

### 推导 3：长度适配可以看成任务条件下的最优推理预算学习

LLaDA-o 最值得记住的一个工程点，是长度适配。设某个任务的最优推理长度为 \(L^\star\)，则作者试图学习一个预算控制机制

\[
L^\star = g(\text{task}, \text{input context}).
\]

这件事的重要性在于：

- 文本理解任务可能不需要很长扩散链；
- 图像生成任务可能需要更长、更细的迭代；
- 若统一模型对所有任务都固定用同样长度，算力会被严重浪费。

因此 LLaDA-o 的长度自适配，本质上是在学习

\[
\text{how much denoising / refinement each task really needs}.
\]

这也说明 unified model 的真正难点之一，不只是共享主干，而是共享后如何按任务分配计算资源。

### 推导 4：为什么说 LLaDA-o 比 MMaDA 更偏“工程可用性”

MMaDA 更像一个范式挑战：diffusion 能否成为统一 backbone？LLaDA-o 则更往前一步，开始回答：

- 如果 diffusion 真能统一，那文本和视觉内部要不要分 branch？
- 解码长度怎么配？
- 推理成本怎么控？

换句话说，LLaDA-o 的贡献不是把 unified diffusion 的抽象理念再重复一遍，而是在告诉我们：

\[
\text{omni-diffusion must solve both heterogeneity and efficiency}.
\]

这也是为什么它比单纯讲 backbone 更接近一个“可用系统”的设计。

## 架构理解

### 1. 为什么它不是简单“MMaDA 的缩写版”

LLaDA-o 和 MMaDA 同属 diffusion-centered unified model 阵营，但两者关注点不一样：

- MMaDA 更强调范式迁移与完整训练栈；
- LLaDA-o 更强调 diffusion family 内部该如何分工，以及长度效率如何解决。

因此它不是简单换个名字再做一遍 diffusion unified model，而是在 omni-diffusion 内部继续细化结构。

### 2. 为什么 Mixture of Diffusion 很关键

如果文本和视觉都硬塞进同一种 diffusion 形式，很可能出现“两边都不舒服”的情况。LLaDA-o 通过 Mixture of Diffusion 承认：

- 统一不一定意味着同构；
- 更合理的 unified model 往往是“共享主干、分支异构”。

这和 Transfusion / Orthus 的思想有相通之处，但 LLaDA-o 把这种异构保留在 diffusion family 内部，而不是走 AR + diffusion 的混合二元结构。

### 3. 为什么长度适配不是纯工程技巧

很多论文会把长度控制当成小优化，但 LLaDA-o 把它放到方法主线上，说明作者意识到：

\[
\text{efficiency is part of model design, not just deployment detail}.
\]

对于 unified model 尤其如此，因为多任务、多模态的长度分布天然非常不均匀。

## 训练流程

LLaDA-o 的训练更适合理解成一个混合 diffusion recipe，而不是一个单一公式。

### 阶段 1：离散理解任务训练 masked diffusion 分支

文本理解、推理相关任务用于训练离散 masked diffusion 分支，让模型在离散符号空间里学会恢复、补全与理解。

### 阶段 2：连续图像任务训练 continuous diffusion 分支

图像生成相关任务用于训练连续 diffusion 分支，让模型在视觉连续状态空间里保留高保真生成能力。

### 阶段 3：混合多模态数据训练共享 backbone 与长度适配

最后再用混合多模态数据把共享 backbone 的跨模态条件能力和长度适配策略一起练起来，使模型不仅“能统一”，而且“统一后推理成本还可控”。

## 直觉 / 理解

我对 LLaDA-o 的理解是：它像 omni-diffusion unified model 从“范式宣言”迈向“工程系统”的一步。MMaDA 会让人重新思考 unified model 为什么必须围绕 AR-LM；LLaDA-o 则更像在说，既然你已经接受 diffusion-centered 这条路线，那么接下来你就必须认真处理：

- 模态异构，
- 长度异构，
- 推理预算异构。

也就是说，它不只在谈“是否统一”，而是在谈“统一之后如何高效地活下去”。

## 与相邻方法的关系

### 对比 MMaDA

MMaDA 更强调“diffusion 能否成为 unified backbone”以及完整后训练栈。LLaDA-o 更强调 omni-diffusion 内部的机制拆分与长度效率，因此两者很像同一阵营里的上下游工作。

### 对比 Transfusion / Orthus

Transfusion、Orthus 通过共享主干加模态专属动力学来缓解异构。LLaDA-o 的相似点是也承认异构不可避免；不同在于，它把这种异构保留在 diffusion family 内部，而不是走 AR + diffusion 混合结构。

### 对比 Show-o2

Show-o2 通过 language head + flow head 统一文本、图像和视频，更像“共享主干，输出动力学分化”。LLaDA-o 则更接近“统一 diffusion 主干，内部再做离散 / 连续 diffusion 分工”。

### 对比 Emu3

Emu3 把统一压缩到单一 next-token prediction；LLaDA-o 则几乎站在相反方向，认为 unified model 内部完全可以保留不同 diffusion 子机制，只要共享 backbone 和计算组织足够好。

## 重要细节

- Architecture: Mixture of Diffusion + shared efficient attention backbone + length adaptation
- Objective: 文本离散 masked diffusion + 图像连续 diffusion + 长度适配训练
- Design: discrete/continuous diffusion heterogeneity under one shared backbone
- Data: 多模态理解、图像生成和长度分布多样的混合任务数据
- Evaluation: multimodal understanding、text-to-image generation、efficiency-oriented omni benchmarks
- Strengths: 对 omni-diffusion 内部结构刻画更细；效率意识很强；比只讲 backbone 更接近可用系统
- Limitations: diffusion 路线整体仍较新；文本侧工具链和用户体验尚不如 AR 模型成熟；长度适配的通用性仍待更多验证

## 我的笔记 / 开放问题

### 我的笔记

我觉得 LLaDA-o 的价值，在于它把“统一”从抽象概念推进到了“不同 diffusion 子机制如何共存、计算如何分配”的层面。很多 unified model 论文容易停在“我们也统一了”，但 LLaDA-o 更像在问：统一之后，不同任务长度差异这么大，你怎么让系统不浪费算力？

这使它很像 omni-diffusion 方法走向工程可用化的一步，而不是单纯再做一个更大的统一 backbone。

### 开放问题

- 这种长度自适应是否会成为 unified model 的通用组件，而不只属于 diffusion 阵营？
- discrete masked diffusion 在文本理解上的上限，能否真正逼近成熟 AR-LM？
- 如果视频也被纳入，长度适配会不会从“有用技巧”变成“绝对必要的核心组件”？
- omni-diffusion 路线最终会不会因为效率问题，反而重新收敛到某种 hybrid 结构？

## 相关笔记

- [MMaDA 笔记](./mmada-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)
- [Orthus 笔记](./orthus-notes.md)

## 参考资料

- You et al., "LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model", arXiv, 2026. https://arxiv.org/abs/2603.01068
- Official repository. https://github.com/ML-GSAI/LLaDA-o
- Hugging Face paper page. https://huggingface.co/papers/2603.01068

