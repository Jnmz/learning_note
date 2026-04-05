# TUNA 笔记

## 元信息

- Topic: unified-models
- Status: draft
- Last updated: 2026-04-05
- Source type: paper
- Primary references:
  - TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models
  - TUNA project page

## 一句话总结

TUNA 的核心主张是：原生统一多模态模型（native unified multimodal model）要想同时把理解和生成都做好，关键不只是“共享一个 transformer 主干”，而是先构造一个真正统一的连续视觉表示空间；为此它把 VAE encoder 与 pretrained representation encoder 串接起来，把视觉 token 同时做成“可生成的 latent”与“可理解的语义表征”。

## 背景 / 问题设定

这篇工作瞄准的是 unified multimodal model 里一个很具体、但又很根本的冲突：

1. 理解任务通常偏好高层语义特征，最好来自强大的判别式视觉 encoder。
2. 生成任务通常偏好连续、可逆、便于 flow / diffusion 建模的 latent。
3. 如果两边各用一套表示，就会出现 representation mismatch：理解分支和生成分支虽然共享 LLM，但它们进入 LLM 的“视觉对象”其实不是同一种东西。

这会带来几个后果：

- 参数共享变得不彻底，因为共享主干看到的是两种格式不同、统计性质不同的视觉输入。
- 训练时理解与生成不一定互相促进，反而可能互相拉扯。
- any-to-any 的数据流会更复杂，因为理解路径和生成路径需要不同的视觉前后处理。

TUNA 的判断是：统一模型的关键瓶颈，在很大程度上不在 decoder，而在 visual representation。也就是说，若视觉表征空间本身不统一，那么“统一模型”常常只是共享了后半段主干，却没有真正统一输入和输出接口。

## 记号

设：

- 输入图像或视频为 \(v\)
- VAE encoder 为 \(E_{\text{vae}}\)
- pretrained representation encoder 为 \(E_{\text{rep}}\)
- 统一视觉 token 序列为 \(u\)
- 文本 token 序列为 \(x = (x_1,\dots,x_T)\)
- 共享 LLM decoder 为 \(f_\theta\)
- 语言头为 \(h_{\text{lm}}\)
- flow matching head 为 \(h_{\text{flow}}\)
- 视觉生成中的噪声时刻为 \(t \in [0,1]\)
- 真实视觉 latent 为 \(z_1\)，噪声样本为 \(z_0\)

文中最关键的结构关系可以抽象写成

\[
z = E_{\text{vae}}(v), \qquad
u = E_{\text{rep}}(z).
\]

这里的 \(z\) 仍保留“可生成”的连续 latent 属性，而 \(u\) 则是进一步被语义化、上下文化后的统一视觉表示。

## 核心思想

### 1. 用级联视觉编码器构造 unified visual representation

TUNA 不是直接选边站：

- 既不只用 VAE latent，因为那样理解能力通常不够强；
- 也不只用判别式视觉表征，因为那样生成路径往往不自然。

它采用 cascade：

```text
image / video
   -> VAE encoder
   -> continuous visual latent
   -> representation encoder
   -> unified visual tokens
   -> LLM decoder
```

因此它的统一不是“强行让一种表示兼顾一切”，而是把可生成性和可理解性沿着同一条视觉管线串联起来。

### 2. 主干统一，但输出动力学分离

TUNA 的共享主干仍然是 LLM decoder。统一视觉 token 与文本 token 一起送入 decoder，然后分成两种输出路径：

- 理解任务：走 language modeling head，做 autoregressive text generation；
- 生成与编辑任务：走 flow matching head，在连续视觉空间里做速度场预测。

这个设计和 Show-o2、Orthus 一样，体现了一个越来越明确的趋势：统一模型的“统一”主要发生在共享上下文建模阶段，而不是要求所有模态共用同一种输出分布。

### 3. 理解与生成要共享同一个视觉对象

TUNA 最重要的 claim 不是“我们也用了 flow matching”，而是：

\[
\text{understanding tokens} \equiv \text{generation tokens}.
\]

这里的“相等”不是数值完全相同，而是指两种任务进入共享 decoder 时使用的是同一视觉表示空间中的 token。这样一来：

- 理解梯度能直接塑造生成所依赖的视觉表示；
- 生成梯度也能反向推动视觉表示保留更多可还原信息；
- 多任务训练更可能出现 mutual benefit，而不是接口层面的彼此隔离。

## 一个简单示意图

```text
image/video --> VAE encoder --> latent z --> representation encoder --> unified visual tokens u
                                                                          |
text tokens ---------------------------------------------------------------+
                                                                          |
                                                              shared LLM decoder
                                                                 /        \
                                                                /          \
                                                       LM head /            \ flow head
                                                              /              \
                                                   text output                visual generation / editing
```

## 详细推导

### 推导 1：为什么 TUNA 的视觉表示是“级联统一”而不是“并列拼接”

TUNA 处理视觉输入的第一步是

\[
z = E_{\text{vae}}(v).
\]

如果只停在这里，那么 \(z\) 更像生成模型偏好的 latent：它连续、压缩、容易配合 flow / diffusion 建模，但未必足够语义化，不一定最适合理解任务。

于是论文进一步引入表示编码器：

\[
u = E_{\text{rep}}(z).
\]

这个公式有两个隐含约束。

第一，\(u\) 不是脱离生成空间另起炉灶的视觉 embedding，而是从 \(z\) 出发构造出来的：

\[
u = E_{\text{rep}}(E_{\text{vae}}(v)).
\]

所以理解路径和生成路径仍然依赖同一个底层连续 latent，而不是两套互不相干的视觉前端。

第二，若我们把“理想统一表征”看成同时优化理解目标与生成目标，那么 TUNA 等价于在寻找

\[
u^\star
=
\arg\min_u
\bigl(
\mathcal{L}_{\text{understand}}(u)
+
\lambda \mathcal{L}_{\text{generate}}(u)
\bigr),
\]

同时通过参数化约束

\[
u = E_{\text{rep}}(E_{\text{vae}}(v))
\]

把这个统一表征限制在“从可生成 latent 出发再语义化”的结构族里。

这和 decoupled 方法本质不同。若 decoupled 方法分别构造

\[
u_{\text{understand}} = E_{\text{understand}}(v), \qquad
u_{\text{generate}} = E_{\text{generate}}(v),
\]

那么共享主干接收到的其实是两种不同分布的视觉 token。TUNA 则试图把它们收束到一个空间 \(u\) 里。

### 推导 2：理解任务仍然是标准的自回归分解

对理解任务，设目标文本为 \(y = (y_1,\dots,y_N)\)，条件是文本前缀和统一视觉 token \(u\)。那么标准自回归分解为

\[
p_\theta(y \mid x, u)
=
\prod_{i=1}^{N} p_\theta(y_i \mid y_{<i}, x, u).
\]

对应负对数似然损失为

\[
\mathcal{L}_{\text{lm}}
=
- \sum_{i=1}^{N} \log p_\theta(y_i \mid y_{<i}, x, u).
\]

这一点看似普通，但对 TUNA 很关键。它说明 TUNA 并没有为了做统一而放弃 LLM 的标准接口。统一视觉表示 \(u\) 的作用，是让 decoder 在保持语言建模形式不变的前提下，能够直接把视觉上下文纳入自回归条件中。

换句话说，TUNA 保持的是

\[
\text{text head} = \text{standard LM head},
\]

而不是重新发明一套理解目标。这让它可以最大化继承预训练 LLM 的语言能力。

### 推导 3：生成任务在统一视觉空间里做 flow matching

对生成任务，TUNA 的项目页与公开摘要都表明它在统一视觉 token 上使用 flow-matching-based visual generation。用标准 flow matching 记号，可以把这一过程写成：

\[
z_t = (1-t) z_0 + t z_1,
\qquad t \in [0,1],
\]

其中 \(z_1\) 是目标视觉 latent，\(z_0 \sim \mathcal{N}(0,I)\) 是噪声起点。对 \(t\) 求导得到目标速度

\[
\frac{d z_t}{dt} = z_1 - z_0.
\]

若 flow head 预测速度场

\[
v_\theta(z_t, t, c),
\]

其中 \(c\) 表示文本条件、历史上下文以及统一视觉 token 提供的条件信息，那么标准 flow matching 损失写成

\[
\mathcal{L}_{\text{flow}}
=
\mathbb{E}_{z_0, z_1, t}
\left[
\left\|
v_\theta(z_t, t, c) - (z_1 - z_0)
\right\|_2^2
\right].
\]

TUNA 的关键不是单纯套用这个 loss，而是让生成 loss 的梯度能反向流经统一视觉表示模块：

\[
h_{\text{flow}} \circ f_\theta \circ E_{\text{rep}} \circ E_{\text{vae}}.
\]

这意味着表示编码器 \(E_{\text{rep}}\) 不只是服务理解，也要为生成优化；它必须把 VAE latent 变成既保语义、又不破坏可生成性的视觉 token。

### 推导 4：联合训练为什么可能是互补而不是冲突

TUNA 强调，在 unified setting 下，understanding data 和 generation data 可以 mutual benefit。把总目标写成

\[
\mathcal{L}
=
\mathcal{L}_{\text{lm}}
+
\alpha \mathcal{L}_{\text{flow}}
+
\beta \mathcal{L}_{\text{edit}}
\]

会更容易看清这一点。

若使用 decoupled 表示，那么对理解路径主要更新的是 \(E_{\text{understand}}\)，对生成路径主要更新的是 \(E_{\text{generate}}\)。它们对共享主干的帮助只能在较后层汇合。

而在 TUNA 中，理解与生成都同时更新

\[
E_{\text{rep}}, \quad f_\theta,
\]

因此参数共享更早、更深。对统一视觉表示 \(u\) 而言，这相当于同时施加两类约束：

\[
u \text{ 要足以支持语义判别},
\qquad
u \text{ 也要足以支持视觉重建与生成轨迹学习}.
\]

如果这两类约束在表示层面是互补的，那么联合训练就会把 \(u\) 推向一个更稳健的共享空间；这正是 TUNA 想要证明的现象。

## 训练配方

基于项目页公开图示、论文摘要，以及可访问的公开解读信息，TUNA 采用分阶段训练思路。下面这部分训练流程描述中，阶段划分属于我对公开材料的一致性整理；若后续拿到正文，可再按论文原文补齐更精确的 batch 配比和数据切换细节。

### 阶段 1：先把统一视觉表示与生成头接到 LLM 上

这一阶段的直觉是：

- LLM 已有语言知识；
- 新加入的 representation encoder 和 flow head 还没有学会如何在统一视觉空间里协同工作；
- 因此先让视觉表示与生成头稳定下来，再全面联合训练。

从优化角度看，这相当于先把“视觉接入问题”单独解决。

### 阶段 2：解冻共享 decoder，做多任务继续预训练

当统一视觉表示可用之后，再让整个模型一起看理解、生成、编辑、视频等多种数据。这个阶段真正建立的是 native unified behavior：不同任务不再只是共用参数，而是共用同一种视觉对象和同一种上下文接口。

### 阶段 3：用高质量数据做 supervised finetuning

最后再用更干净、更高质量的数据做 SFT，把 instruction following、editing、image/video generation 等能力往实用方向收紧。

## 架构理解

### 1. 为什么说 representation encoder 是 TUNA 的胜负手

如果没有 representation encoder，那么 VAE latent 更偏“可生成性”，很难直接达到强视觉理解表现；如果只用 representation encoder 而不经过 VAE latent，又难以和流式视觉生成对齐。

所以 TUNA 的关键不是“两个 encoder 堆起来”这么简单，而是它把二者串成一条有方向的信息路径：

\[
\text{visual signal}
\rightarrow
\text{generative latent}
\rightarrow
\text{semantic refinement}
\rightarrow
\text{shared decoder}.
\]

这个顺序非常重要，因为它保证最终统一 token 仍锚定在生成空间附近，而不是彻底偏向纯判别特征。

### 2. 为什么 unified representation 比 dual representation 更自然

若理解和生成各自有一套视觉 token，那么模型内部就不得不学习一个隐式翻译器：

\[
u_{\text{understand}}
\leftrightarrow
u_{\text{generate}}.
\]

这会让共享 decoder 的一部分容量被迫用于“对齐两种视觉语言”。TUNA 试图直接把这种隐式翻译前移并显式消掉，也就是让模型从一开始就只说一种视觉语言。

### 3. 为什么它属于 native unified model

我理解“native”在这里不是营销词，而是强调：

- 理解不是外接独立 MLLM；
- 生成不是外接独立 diffusion backbone；
- 两者在同一个视觉表示空间、同一个 decoder 上完成。

这比“两个系统拼接后共享提示词格式”的统一程度更高。

## 与相邻方法的关系

### 对比 Show-o / Show-o2

Show-o 系列也追求 native unified model，但其焦点更偏向“统一 decoder 如何同时处理文本和视觉生成”。TUNA 则把焦点进一步前移到视觉表示空间本身，强调若 visual representation 不统一，decoder 再统一也仍有接口裂缝。

### 对比 Orthus / Transfusion

Orthus、Transfusion 都体现了“共享主干 + 专属输出动力学”的思路。TUNA 与它们相似之处在于也把文本输出与视觉生成分开；但 TUNA 更突出 unified visual representation 的必要性，而不只是强调输出头分工。

### 对比传统 decoupled MLLM + generator

传统方案通常是：

- 理解靠强大的 vision encoder + LLM；
- 生成靠 text-to-image / text-to-video generator；
- 中间通过 prompt、caption 或 latent adapter 松耦合。

TUNA 则试图把这些接口内化到同一个模型里，因此它更像一个“统一系统”，而不是系统编排。

## 重要细节

- Architecture: VAE encoder + pretrained representation encoder + shared LLM decoder + LM head + flow matching head
- Representation: 连续 unified visual tokens，同时服务理解、生成、编辑与视频任务
- Objective: 文本自回归 next-token prediction 与视觉 flow matching 联合训练
- Modalities: image、video、text，支持 understanding、generation、editing
- Claimed advantage: 避免 decoupled representation 带来的格式错配，使理解与生成互相促进
- Empirical message: 更强的 pretrained representation encoder 会系统性提升各类任务表现

## 我的笔记 / 开放问题

### 我的笔记

我觉得 TUNA 真正有意思的地方，是它把 unified multimodal model 的讨论从“decoder 能不能统一”推进到了“visual representation 应该如何统一”。这比简单讨论用 AR 还是 diffusion 更基础，因为很多训练冲突其实在视觉 token 进入主干之前就已经埋下了。

另一个值得记住的点是，它没有把统一理解成“所有模态都必须长成同一种离散 token”。TUNA 接受视觉空间应该保持连续，也接受视觉表征需要专门的 representation encoder 去语义化；它追求的是接口统一，而不是形式上的完全同构。

### 开放问题

- representation encoder 到底保留了多少 VAE latent 的可逆细节，多少已经转成更偏语义的判别特征？
- 当模型规模继续增大时，统一表示是否还优于“理解强 encoder + 生成强 latent”这种弱耦合方案？
- 在视频场景里，统一视觉表示对长时序一致性的帮助，究竟主要来自共享空间，还是主要来自更大规模联合训练？
- flow head 与 language head 的梯度是否会在某些阶段重新产生竞争，需要更细的 loss balancing 或 routing？

## 相关笔记

- [统一多模态模型总览](./unified-multimodal-models-overview.md)
- [Orthus 笔记](./orthus-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)

## 参考资料

- Liu et al., "TUNA: Taming Unified Visual Representations for Native Unified Multimodal Models", arXiv, 2025. https://arxiv.org/abs/2512.02014
- TUNA project page. https://tuna-ai.org/
- Hugging Face paper page. https://huggingface.co/papers/2512.02014
