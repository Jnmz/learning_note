# Show-o Notes

## Metadata

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Show-o: One Single Transformer to Unify Multimodal Understanding and Generation
  - Show-o project page and official repository

## One-Sentence Takeaway

Show-o 的核心贡献不是“把图像 token 塞进 LLM”这么简单，而是把文本的自回归建模和图像 token 的离散扩散建模放进同一个 transformer 里，让一个骨干同时承担理解和生成。

## Background / Problem Setup

在 Show-o 之前，统一多模态模型大致有两条路：

1. 用一个理解模型负责问答，再外挂一个扩散模型负责生成。
2. 把文本和图像都离散化成 token，然后统一做 autoregressive next-token prediction。

第一条路的问题是系统被拆成多个专家模块，接口复杂，训练和推理都不够“原生统一”。第二条路的问题是，如果图像也严格按自回归逐 token 生成，那么高分辨率图像会带来很长的采样链，速度和质量都会吃亏。

Show-o 想做的事情更激进一些：

- 文本仍然沿用 LLM 最擅长的自回归建模。
- 图像不强行改成纯自回归，而是采用离散扩散式的 mask-token prediction。
- 两者共享同一个 transformer 主干，只在注意力模式和训练目标上做区分。

从这个角度看，Show-o 不是简单的 unified tokenizer，而是 unified backbone + mixed objective。

## Notation

设：

- 文本 token 序列为 \(v = (v_1, \dots, v_T)\)
- 图像经 tokenizer 离散化后的 token 序列为 \(u = (u_1, \dots, u_N)\)
- 统一提示模板后的输入序列记为 \(s\)
- mask 后的图像 token 序列记为 \(\tilde{u}\)
- 模型参数记为 \(\theta\)
- 文本的 next-token prediction loss 记为 \(\mathcal{L}_{\text{NTP}}\)
- 图像的 mask-token prediction loss 记为 \(\mathcal{L}_{\text{MTP}}\)
- 总损失记为 \(\mathcal{L}\)

Show-o 里的特殊 token 包括任务标记 `[MMU]`、`[T2I]`，以及文本边界 `[SOT]` / `[EOT]`、图像边界 `[SOI]` / `[EOI]`。

## Core Idea

### 1. Shared Discrete Space

Show-o 基于预训练 LLM，因此它优先选择把多模态统一到离散序列空间里。文本直接使用 LLM 原生 tokenizer，图像则使用类似 MAGVIT-v2 的离散 tokenizer，把一张图映射成有限词表中的图像 token 序列。

这样做的直接收益是：

- 模型输入输出都能写成“序列”
- 不同任务可以统一为“给定前文，预测目标 token 或目标 token 分布”
- 统一 prompting 成为可能

### 2. One Backbone, Two Modeling Modes

Show-o 的主干仍然是 decoder-only transformer，但它不要求所有 token 都遵循同一种生成规律：

- 文本 token 用 causal attention，自回归地产生
- 图像 token 用 full attention，以 mask prediction 的方式并行去噪

所以统一点不在于“所有 token 都按 next-token 生成”，而在于“所有 token 都进入同一个主干，被同一组参数理解，只是遵循不同的条件依赖结构”。

### 3. Omni-Attention

这是 Show-o 里很关键的接口设计。它允许：

- 文本 token 看到前面的图像 token
- 图像 token 看到前面的文本 token
- 图像 token 之间是 full attention
- 纯文本场景时退化成标准 causal attention

于是同一个模型既能做 image-conditioned text generation，也能做 text-conditioned image generation。

## A Simple Diagram

```text
image/text input
    |
    v
tokenizers -> unified prompt sequence
    |
    v
one transformer backbone
    |
    +--> text tokens: causal attention + next-token prediction
    |
    +--> image tokens: full attention + masked denoising prediction
    |
    v
answer text / generated image tokens
```

## Detailed Derivation

### Derivation 1: Text Objective Is Standard Conditional Language Modeling

对于多模态理解任务，Show-o 的目标通常是“给定图像 token 和文本前缀，预测答案文本”。这本质上仍然是条件语言模型：

\[
p_\theta(v \mid u)
= \prod_{t=1}^{T} p_\theta(v_t \mid u, v_{<t}).
\]

对数似然可写成

\[
\log p_\theta(v \mid u)
= \sum_{t=1}^{T} \log p_\theta(v_t \mid u, v_{<t}).
\]

训练时最大化对数似然，等价于最小化负对数似然：

\[
\mathcal{L}_{\text{NTP}}
= - \sum_{t=1}^{T} \log p_\theta(v_t \mid u, v_{<t}).
\]

这一步没有任何新奇之处，但它很重要，因为 Show-o 明确不打算推翻 LLM 在文本建模上的成功经验。它保留了自回归语言建模这一部分，把“统一”的难点放在图像侧。

进一步看，如果输入里根本没有图像，式子自然退化为普通 LLM 训练目标：

\[
\mathcal{L}_{\text{NTP}}
= - \sum_{t=1}^{T} \log p_\theta(v_t \mid v_{<t}).
\]

所以 Show-o 不是另起炉灶，而是在标准 LM 目标外面叠加一个视觉生成目标。

### Derivation 2: Why Masked Image Modeling Can Be Viewed As Discrete Diffusion

Show-o 图像侧采用的是离散扩散式建模，具体落地成 MaskGIT 风格的 mask token prediction。其逻辑可以一步步写清楚。

先从干净图像 token 序列 \(u\) 出发。训练时，随机采样一个 mask ratio \(r \sim p(r)\)，再采样一个 mask 集合 \(M \subseteq \{1,\dots,N\}\)，满足大约有 \(rN\) 个位置被遮住。于是得到 corrupted 序列

\[
\tilde{u}_i =
\begin{cases}
[\text{MASK}], & i \in M, \\
u_i, & i \notin M.
\end{cases}
\]

模型需要根据未被 mask 的图像 token、前面的文本条件，以及统一序列结构，恢复被遮住的位置。因此条件分布可以写成

\[
p_\theta(u_M \mid u_{\bar{M}}, v),
\]

其中 \(u_M\) 表示被 mask 的位置，\(u_{\bar{M}}\) 表示未被 mask 的位置。

如果只对被 mask 的位置计算似然，那么训练目标就是

\[
\mathcal{L}_{\text{MTP}}
= - \mathbb{E}_{M \sim p(M)}
\left[
\sum_{i \in M} \log p_\theta(u_i \mid \tilde{u}, v)
\right].
\]

为什么这能被称为“离散扩散”？

因为这里的 corruption 过程不是给连续变量加高斯噪声，而是把离散 token 逐步替换成 `[MASK]`。随着 mask 比例增大，序列的信息逐步退化；随着模型逐步恢复这些位置，序列逐步回到数据分布。这个“从数据分布走向高熵分布，再逆向恢复”的思想，与扩散模型是一致的，只是噪声机制从高斯噪声改成了离散 mask corruption。

换句话说：

- 连续扩散里，前向过程是加噪
- Show-o 的离散扩散里，前向过程是加 mask
- 连续扩散里，逆过程是预测噪声或 score
- Show-o 里，逆过程是预测被遮住 token 的类别分布

这也是 Show-o 能在同一个 transformer 中统一文本和图像的关键，因为图像生成不再依赖独立的 U-Net 或额外文本编码器。

### Derivation 3: The Joint Objective Is A Weighted Mixture Of Two Factorizations

Show-o 最终训练的是文本目标和图像目标的联合：

\[
\mathcal{L}
= \mathcal{L}_{\text{NTP}} + \lambda \mathcal{L}_{\text{MTP}}.
\]

这个式子表面上简单，但它反映了一个更深的建模分解：

一方面，文本部分建模的是

\[
p_\theta(v \mid u),
\]

另一方面，图像部分建模的是

\[
p_\theta(u_M \mid u_{\bar{M}}, v).
\]

因此 Show-o 并不是直接拟合单一的联合分布 \(p(u,v)\) 的一种唯一 factorization，而是在不同任务格式下切换条件分解方式：

- 理解任务时，更像在拟合 \(p(v \mid u)\)
- 生成任务时，更像在拟合离散去噪链上的条件分布

从优化角度看，这意味着同一个 backbone 同时承受两类不同的约束：

1. 语言侧要求因果依赖和顺序推理能力。
2. 视觉侧要求全局一致的并行重建能力。

这也解释了为什么 omni-attention 很重要。若图像仍然使用 causal attention，那么 \(\mathcal{L}_{\text{MTP}}\) 的 full-context 重建优势就会丢失；若文本也改成 full attention，则预训练 LLM 的因果语言能力会被破坏。

## Training Pipeline

论文把训练分成三步，这和方法本身一样重要。

### Stage 1: Learn Image Token Embeddings And Pixel Dependency

由于图像 token 的词嵌入是新加进去的，模型一开始并不理解这些新符号。所以第一阶段的重点是：

- 学会图像 token embedding
- 学会图像 token 之间的依赖关系
- 初步建立图文对齐

这一阶段同时保留文本语料训练，以尽量不伤害已有语言能力。

### Stage 2: Strengthen Image-Text Alignment

在有了基础视觉生成能力后，再进一步用大规模图文数据强化 text-to-image 和 captioning 的对齐。

### Stage 3: High-Quality Fine-Tuning

最后用高质量筛选数据和 instruction-style 数据做精调，让模型更适合实际的 multimodal understanding、mixed-modality generation，以及更稳定的视觉生成。

这三步实际上对应三个难点：

- 先学“看懂新词表”
- 再学“文本和图像如何对齐”
- 最后学“如何在任务格式里稳定工作”

## Intuition / Interpretation

我觉得 Show-o 的关键直觉可以概括成一句话：

不要强迫所有模态都服从同一种生成范式，而是让它们共享同一个计算主体，但保留最适合自己的建模方式。

文本适合因果自回归，因为语言天然有顺序结构。图像更适合并行式去噪，因为图像更强调全局一致性。Show-o 通过一个共享 transformer 把这两件事“焊”在一起，而不是强行折叠成纯 AR 或纯 diffusion。

从系统角度看，它是一种中间路线：

- 比“理解模型 + 生成模型”更原生统一
- 比“所有模态都做 next-token prediction”更尊重视觉生成的归纳偏置

## Relation to Other Methods

### Versus Chameleon

Chameleon 更接近“全部离散 token，全部 autoregressive”。Show-o 则坚持图像生成不必完全服从 next-token prediction，因此在采样步数上更有优势。

### Versus LLaVA-Style MLLMs

LLaVA 一类方法很强于理解，但没有原生图像生成能力。Show-o 的贡献是把生成任务也纳入统一主干。

### Versus Janus / Janus-Pro

这类模型同样关注理解与生成统一，但 Show-o 的一个鲜明特点是明确把“文本 AR + 图像离散 diffusion”作为设计中心，而不是只在输入接口上统一。

## Important Details

- Architecture: 以预训练 LLM 为主干，扩展图像 token embedding，并加入 omni-attention
- Objective: 文本用 next-token prediction，图像用 MaskGIT 风格的 mask-token prediction
- Data: 文本语料、ImageNet-1K、约 35M 图文对
- Evaluation: multimodal understanding、text-to-image、inpainting、extrapolation、mixed-modality generation
- Strengths: 一个主干同时做理解和生成，图像采样步数明显少于纯 AR 图像生成
- Limitations: 图像生成仍依赖离散 tokenizer 质量；理解能力对视觉表示选择敏感；统一训练会带来目标冲突

## My Notes / Open Questions

### My Notes

我觉得 Show-o 最值得学的不是某个单独技巧，而是它对“统一”的定义比较克制。它没有执着于形式上完全一致，而是只要求主干统一、接口统一、训练流程尽量统一。

另一个值得注意的点是，论文后续版本明显更重视“离散视觉表示对理解是否足够好”这个问题。这其实触及统一模型里的核心张力：生成友好的表示，不一定是理解友好的表示。

### Open Questions

- 如果图像 tokenizer 更强，Show-o 的理解能力能否接近主流 MLLM？
- 当图像分辨率继续提高时，离散 token 序列长度是否又会成为瓶颈？
- 文本 AR 与图像 MTP 的联合训练，是否存在更稳定的 curriculum 或 loss balancing 策略？

## See Also

- [Unified Multimodal Models Overview](./unified-multimodal-models-overview.md)
- [Show-o2 Notes](./show-o2-notes.md)

## References

- Xie et al., "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation", arXiv, 2024/2025. https://arxiv.org/abs/2408.12528
- Show-o project page. https://showlab.github.io/Show-o/
- Official repository. https://github.com/showlab/Show-o
