# Emu3 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - Emu3: Next-Token Prediction is All You Need
  - Emu3 official repository

## 一句话总结

Emu3 是统一模型里最纯粹的 autoregressive 路线之一：它坚持把文本、图像、视频都压到同一种离散 token 接口里，再用同一个 decoder-only transformer 做 next-token prediction，从而把理解、图像生成、视频生成和交错续写全部还原成一个统一的序列学习问题。

## 背景 / 问题设定

Emu3 面向的是 unified multimodal model 设计中的一个根本分歧：

1. 一类方法认为，不同模态应保留不同生成动力学，例如文本用 AR、图像用 diffusion、视频再加专门时序模块。
2. 另一类方法认为，只要把不同模态变成同一种 token 接口，自回归序列建模本身就足以覆盖理解和生成。

Emu3 明确站在第二类立场，并且比很多工作走得更彻底。它要验证的不是“多模态大模型能不能工作”，而是更强、更挑衅的问题：

- next-token prediction 是否可以成为文本、图像、视频的统一训练原则？
- 一旦视觉表示也离散化为 token，模态专属目标还有多大必要？
- 如果图像和视频都被视为序列，那么理解、创作和长文档级交错生成是否都能统一到同一种接口中？

因此 Emu3 的研究重点其实不是一个 fancy 模块，而是一个非常强的建模判断：

\[
\text{next-token prediction is all you need}.
\]

这个判断如果成立，意味着 unified model 的主要难点就从“如何为不同模态设计不同 head / objective”转移到了：

- 视觉 tokenizer 是否足够强；
- 序列接口是否足够稳定；
- 长混合序列上的 AR 建模是否足够可扩展。

## 记号

设：

- 文本 token 序列为 \(t = (t_1,\dots,t_m)\)
- 图像或视频输入为 \(x\)
- 视觉 tokenizer 为 \(\tau_{\text{vis}}\)
- 图像或视频经离散化后得到的视觉 token 序列为 \(v = (v_1,\dots,v_n)\)
- 混合多模态序列为 \(s = (s_1,\dots,s_T)\)
- 统一自回归模型为 \(f_\theta\)
- 统一训练损失为 \(\mathcal{L}_{\text{AR}}\)

Emu3 的关键映射关系可以抽象写成

\[
v = \tau_{\text{vis}}(x),
\qquad
s = \operatorname{concat}(\text{text tokens}, \text{visual tokens}, \text{special tokens}).
\]

随后统一模型在 \(s\) 上做标准因果建模：

\[
p_\theta(s) = \prod_{i=1}^{T} p_\theta(s_i \mid s_{<i}).
\]

## 核心思想

### 1. 一切都变成 token

Emu3 的统一方式非常直接，甚至有些“强硬”：

- 文本本来就是 token；
- 图像被视觉 tokenizer 压成离散视觉 token；
- 视频被视为更长、更具结构化的视觉 token 序列；
- 最终所有内容都进入同一个序列世界。

也就是说，Emu3 不是“共享一个主干，但保留不同模态的表示和目标”，而是试图在接口层就把模态差异尽量抹平。

### 2. 一切都做 next-token prediction

和 Show-o、Transfusion、LLaDA-o、MMaDA 这类“共享主干但保留不同动力学”的路线不同，Emu3 刻意追求目标函数极简主义。无论是：

- image understanding，
- text-to-image，
- text-to-video，
- interleaved continuation，

最终都被写成同一种形式：

\[
\mathcal{L}_{\text{AR}}
=
- \sum_{i=1}^{T} \log p_\theta(s_i \mid s_{<i}).
\]

因此 Emu3 的统一不只是共享 backbone，而是连 training objective 也尽量统一。

### 3. 接口统一优先于模态专门化

Emu3 的方法判断是：只要统一 token 接口足够彻底，同一个 transformer 就会自然吸收跨模态条件关系，而不必显式保留图像 diffusion、视频 flow、或其他模态专属输出动力学。

换句话说，它优先相信的是

\[
\text{interface unification}
\rightarrow
\text{capability unification},
\]

而不是

\[
\text{capability unification}
\rightarrow
\text{keep modality-specific objectives}.
\]

## 一个简单示意图

```text
text ------------------------------------+
                                         |
image/video --> visual tokenizer --------+--> unified token sequence --> AR transformer --> next token
```

## 详细推导

### 推导 1：所有任务都可以统一为条件序列建模

Emu3 的第一原则是：一旦多模态输入输出都被转换成序列 \(s\)，那么训练目标就退化成标准语言模型目标

\[
p_\theta(s) = \prod_{i=1}^{T} p_\theta(s_i \mid s_{<i}).
\]

对应负对数似然为

\[
\mathcal{L}_{\text{AR}}
=
- \sum_{i=1}^{T} \log p_\theta(s_i \mid s_{<i}).
\]

这个式子看起来太普通，几乎不像“多模态方法公式”，但它恰恰体现了 Emu3 的核心激进之处：它拒绝为多模态任务再写一套额外的统一数学形式。

关键在于不同任务如何被编译成 \(s\)。

若是多模态理解，例如 image-conditioned QA，可以组织成

\[
s = [v,\, q,\, a],
\]

其中 \(v\) 是图像 token，\(q\) 是问题文本，\(a\) 是答案文本。此时训练模型拟合的是

\[
p_\theta(a \mid v, q)
=
\prod_{j=1}^{|a|} p_\theta(a_j \mid v, q, a_{<j}).
\]

若是 text-to-image，则可组织成

\[
s = [t,\, v],
\]

于是模型拟合

\[
p_\theta(v \mid t)
=
\prod_{j=1}^{n} p_\theta(v_j \mid t, v_{<j}).
\]

若是 text-to-video，也不过是把 \(v\) 换成更长的视频 token 序列。因此 Emu3 的一个核心观察可以写成：

\[
\text{understanding, image generation, video generation}
\subset
\text{conditional continuation}.
\]

### 推导 2：视觉 tokenizer 是把视觉问题转写成语言模型问题的关键映射

Emu3 成立的前提不是 transformer magically 理解像素，而是视觉 tokenizer \(\tau_{\text{vis}}\) 足够强，能把图像或视频映射成高质量离散序列：

\[
v = \tau_{\text{vis}}(x).
\]

这里的关键不是“离散化”三个字本身，而是离散化必须满足两类约束。

第一，表示保真约束：token 序列 \(v\) 不能丢失太多可重建细节，否则视觉生成上限会被 tokenizer 直接卡死。

第二，序列可建模约束：得到的 token 序列不能过于混乱，否则 AR 模型很难学到稳定的条件分布。

理想情况下，我们希望存在某个离散表示 \(v\)，使得原始视觉样本分布 \(p(x)\) 可以通过离散 token 分布 \(p(v)\) 近似表达，再由解码器 \(\tau_{\text{vis}}^{-1}\) 近似恢复。这可以抽象地写成

\[
x
\xrightarrow{\tau_{\text{vis}}}
v
\xrightarrow{\tau_{\text{vis}}^{-1}}
\hat{x},
\qquad
\hat{x} \approx x.
\]

一旦这个近似足够好，Emu3 的整个方法就可以被理解成：

\[
\text{model } p(v \mid \text{context}) \text{ instead of } p(x \mid \text{context}).
\]

也就是说，视觉 tokenizer 实际上承担了“把视觉建模问题改写成语言建模问题”的职责。Emu3 越成功，越说明 tokenizer 才是这条路线真正的隐形主干。

### 推导 3：视频只是更长、更结构化的视觉 token 序列

Emu3 非常重要的一点，是它不只统一 text 和 image，还把 video 一起纳入 next-token prediction 路线。

若视频由 \(F\) 帧组成，每帧离散化后得到一段 token，那么整个视频可以写成一个更长的序列

\[
v^{\text{video}}
=
\bigl(
v^{(1)}_1,\dots,v^{(1)}_{n_1},
v^{(2)}_1,\dots,v^{(2)}_{n_2},
\dots,
v^{(F)}_1,\dots,v^{(F)}_{n_F}
\bigr).
\]

于是 text-to-video 目标并不需要新公式，只是变成

\[
p_\theta(v^{\text{video}} \mid t)
=
\prod_{j=1}^{N}
p_\theta(v^{\text{video}}_j \mid t, v^{\text{video}}_{<j}),
\]

其中 \(N = \sum_f n_f\)。

这个公式有两个重要含义。

第一，Emu3 认为视频和图像的差异首先是“序列长度和结构复杂度”的差异，而不是“必须换一种生成动力学”的差异。

第二，时序一致性在 Emu3 中不是通过额外的 temporal diffusion 或 flow objective 显式保证的，而是通过单一 AR 模型对长 token 前缀的条件建模隐式学习出来的。

因此它对视频的态度其实很清楚：

\[
\text{video} \approx \text{long structured visual text}.
\]

这是一种极强的建模假设，也是 Emu3 最有代表性的地方。

### 推导 4：统一损失的简洁性把所有困难前移到了表示与序列长度

Emu3 的总损失可以写得非常干净：

\[
\mathcal{L}_{\text{AR}}
=
- \sum_{i=1}^{T} \log p_\theta(s_i \mid s_{<i}),
\]

但这个简洁性并不意味着问题变简单了，它只是把难点转移了。

如果我们把总误差粗略分解成

\[
\text{total difficulty}
\approx
\text{tokenization difficulty}
+
\text{long-context modeling difficulty}
+
\text{sampling difficulty},
\]

那么 Emu3 基本上是主动接受：

- 不再在 loss 里处理模态差异；
- 而是把压力集中到视觉离散化质量和超长 AR 建模能力上。

因此它的统一路线其实是一种“问题重分配”：

\[
\text{less objective complexity}
\Rightarrow
\text{more pressure on tokenizer and sequence modeling}.
\]

这也是为什么 Emu3 的成败很大程度取决于视觉 tokenizer、序列长度控制、以及推理阶段的采样效率。

## 架构理解

### 1. 为什么 Emu3 不是简单的“多模态 LLM”

如果只是把视觉特征作为条件塞给 LLM，再让 LLM 输出文本，那是典型 MLLM。Emu3 更激进，因为它还要求：

- 图像输出也由同一个 AR 主干生成；
- 视频输出也由同一个 AR 主干生成；
- 并且这些输出都通过统一 token 序列完成。

因此 Emu3 的 unified ambition 不只是“看图说话”，而是“把视觉生成也视作语言模型问题”。

### 2. 为什么这条路线天然支持 interleaved generation

一旦图像、视频、文本都被编码成同一种序列对象，那么长文档级交错生成就自然变成

\[
[text][image][text][video][text]\dots
\]

这样的续写问题。

也就是说，interleaved generation 在 Emu3 里不是额外外挂能力，而是统一序列建模自然带出来的结果。只要训练数据里存在这类样本，模型就能直接把它们纳入标准 AR 学习。

### 3. 为什么说视觉 tokenizer 是“隐形主干”

表面上 Emu3 最显眼的是单一 AR transformer，但实际上，若 tokenizer 不够好，会出现三类问题：

- 视觉细节丢失，生成上限变低；
- token 序列统计结构差，训练更难；
- 图像和视频长度膨胀，推理代价失控。

所以 Emu3 的一个重要现实是：

\[
\text{unified AR model quality}
\le
\text{visual tokenizer quality ceiling}.
\]

这也是为什么 Emu3 看似在强调“next-token prediction is all you need”，实际上却高度依赖一个足够强的视觉前端。

## 训练流程

Emu3 的训练目标很统一，但 recipe 仍然不简单。根据论文和公开材料，它的训练重点更像是围绕统一 token 接口组织多源数据。

### 阶段 1：建立文本与视觉 token 的共同词表生态

模型首先要学会处理新加入的视觉 token，理解这些 token 在统一序列中的统计角色。这一步并不只是“识别新词”，更是在建立：

- 文本 token 和视觉 token 的共存分布；
- 图文边界符与模板标记的作用；
- 视觉 token 之间的基本局部依赖。

### 阶段 2：用图文和视频数据强化条件生成

当模型已经接受“视觉也是 token 序列”后，再用 image-text、video-text、以及理解数据强化条件分布学习，让它真正掌握：

- \(p(\text{text} \mid \text{image/video})\)
- \(p(\text{image} \mid \text{text})\)
- \(p(\text{video} \mid \text{text})\)

这些关系。

### 阶段 3：用交错长序列对齐 unified behavior

最后再用 interleaved multimodal 数据把不同能力缝到同一长上下文里，让模型学会在一个序列中完成：

- 读图回答，
- 文生图，
- 文生视频，
- 图文视频混合续写。

这一步对 Emu3 特别重要，因为它的方法价值不只在单任务，而在“统一接口是否真的能支撑复杂混合交互”。

## 直觉 / 理解

我对 Emu3 的理解是：它相当于给 unified model 领域做了一次最彻底的 AR 极限测试。

很多方法会说“统一当然重要，但图像、视频还是应该保留自己的动力学”。Emu3 的回答是：先别急着保留，把视觉都写成 token，再看看单一 NTP 到底能走多远。

这条路线的美感很强，因为它在概念上极其整齐：

- 一个 tokenizer 体系，
- 一个序列接口，
- 一个 transformer 主干，
- 一个 next-token objective。

但它的代价也同样集中：如果视觉 token 不够高效，或者序列太长，那么这种整齐会直接变成训练与采样负担。

## 与相邻方法的关系

### 对比 Chameleon

Chameleon 也是典型的 token-unified AR 路线，Emu3 与它最像。两者都高度信任“视觉也只是 token”这一观点。Emu3 的进一步推进在于把视频也更明确地纳入同一叙事。

### 对比 Show-o

Show-o 选择共享 transformer 主干，但图像生成仍然使用离散去噪式 mask prediction。Emu3 则连这一步也不保留，要求图像生成也服从 AR next-token prediction，因此统一程度更高，但也更依赖 token 序列质量与采样效率。

### 对比 Transfusion / Orthus

Transfusion、Orthus 都体现“共享主干 + 模态专属输出动力学”的思路。Emu3 正好站在对面：它尽量不保留专属输出动力学，而是把所有困难都塞回统一 AR 接口中。

### 对比 MMaDA / LLaDA-o

MMaDA、LLaDA-o 倾向于把统一建立在 diffusion family 上。Emu3 则代表另一种完全不同的信念：统一不一定要围绕 diffusion，也可以围绕序列建模本身建立。

## 重要细节

- Architecture: 视觉 tokenizer + unified token sequence + 单一 decoder-only AR transformer
- Objective: 文本、图像、视频统一的 next-token prediction
- Modalities: text、image、video，以及它们的交错长序列
- Capability framing: understanding、text-to-image、text-to-video、interleaved multimodal continuation
- Strengths: 结构和目标极度统一；接口非常整齐；天然适合统一长上下文建模
- Limitations: 严重依赖视觉离散化质量；图像和视频 AR 采样代价高；很多统一代价被转移到 tokenizer 与长序列建模

## 我的笔记 / 开放问题

### 我的笔记

我觉得 Emu3 最有价值的地方，是它把 unified model 的“统一程度”推进到了一个几乎没有借口的程度。很多方法都说自己统一，但往往统一的是 backbone，不是 objective；统一的是理解，不是生成；统一的是图像，不是视频。Emu3 则几乎把这些退路都堵上了。

但也正因为如此，它暴露的问题特别真实：如果纯 AR 统一路线最终受限，很可能不是因为 unified idea 错了，而是因为视觉 tokenizer 和长序列采样在工程上太昂贵。换句话说，Emu3 像是一个很干净的对照实验，帮助我们看清 unified AR 路线真正的瓶颈在哪里。

### 开放问题

- 视觉 tokenizer 的提升，究竟会先改善生成保真，还是先改善统一训练稳定性？
- 当图像和视频分辨率继续提高时，AR 采样成本会不会成为纯 AR unified model 的根本硬上限？
- 视频时序一致性是否真的能仅靠长序列 AR 隐式学好，还是迟早需要显式 temporal inductive bias？
- 若未来出现更高效的视觉 token 压缩，Emu3 这条“纯 AR 统一路线”是否会重新变得更有竞争力？

## 相关笔记

- [Chameleon 笔记](./chameleon-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)

## 参考资料

- Wang et al., "Emu3: Next-Token Prediction is All You Need", arXiv, 2024. https://arxiv.org/abs/2409.18869
- Official repository. https://github.com/baaivision/Emu3
- Hugging Face paper page. https://huggingface.co/papers/2409.18869

