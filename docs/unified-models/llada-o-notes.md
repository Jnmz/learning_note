# LLaDA-o 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model
  - LLaDA-o official repository

## 一句话总结

LLaDA-o 的核心贡献是把文本理解所需的离散 masked diffusion 与视觉生成所需的连续 diffusion 放进一个 Mixture of Diffusion 框架里，并进一步通过 length adaptation 让统一模型在不同模态长度下更灵活地解码。

## 背景 / 问题设定

LLaDA-o 针对的是 omni-diffusion 模型里的两个痛点：

- 文本理解和视觉生成需要的 diffusion 形式并不相同
- 多模态解码长度差异很大，如果架构对长度不够适配，就会出现明显冗余和效率问题

因此它不是简单追求“全模态都 diffusion”，而是试图更精细地组织不同 diffusion 子机制。

## 记号

设：

- 文本序列为 \(x\)
- 图像连续变量为 \(y\)
- 共享注意力主干为 \(f_\theta\)
- 离散 masked diffusion 目标为 \(\mathcal{L}_{\text{disc}}\)
- 连续 diffusion 目标为 \(\mathcal{L}_{\text{cont}}\)

## 核心思想

### 1. Mixture of Diffusion

LLaDA-o 的 MoD 框架把两类扩散机制解耦：

- 文本理解走离散 masked diffusion
- 视觉生成走连续 diffusion

但二者并不各自为政，而是通过共享的 attention backbone 耦合在一起。

### 2. 共享高效注意力主干

论文强调这个主干是 simple and efficient 的，并试图减少 fixed conditions 下的重复计算。这说明它不仅关心统一，还关心统一之后的实际计算效率。

### 3. Length Adaptation

LLaDA-o 另一个很新颖的点是长度自适应。它通过数据驱动策略，让模型在不同模态和不同任务下能自然适配不同解码长度，而不需要改架构。

## 关键机制

### Discrete Understanding + Continuous Generation

这和 Show-o、Transfusion 有相似精神，但 LLaDA-o 明确把这种混合关系重新表述成 diffusion 内部的“混合体”，而不是“语言模型 + 视觉生成器”的外部拼接。

### Shared Backbone Coupling

不同 diffusion 分支通过共享注意力主干耦合，统一信息流仍然发生在一个共同的上下文处理核心中。

### Data-Centric Length Adaptation

长度自适应的价值在于，统一模型往往面临：

- 文本输出长、视觉输出短或相反
- 条件长度与生成长度分布不一致

如果没有长度适配，统一解码容易浪费算力并影响稳定性。

## 直觉 / 理解

LLaDA-o 像是在把“统一 diffusion 模型”做得更工程化、更可用。它不只关心理论上能统一，还关心统一之后在长度、计算和多模态接口上是否真的顺手。

## 与其他方法的关系

### 对比 MMaDA

MMaDA 更强调 diffusion 统一大模型的完整训练范式与后训练；LLaDA-o 更强调 MoD 架构本身与长度自适应。

### 对比 Show-o / Transfusion

三者都属于“不同模态保留不同动力学”的家族，但 LLaDA-o 是更明确的 omni-diffusion 表述。

### 对比 Emu3

Emu3 走极致 AR；LLaDA-o 则走混合 diffusion。两者几乎代表当前统一模型两端最鲜明的哲学分歧。

## 重要细节

- Architecture: Mixture of Diffusion + shared attention backbone
- Objective: 文本离散 masked diffusion + 视觉连续 diffusion
- Data: 多模态理解与图像生成数据
- Evaluation: multimodal understanding、text-to-image generation、omni-diffusion benchmarks
- Strengths: 统一 diffusion 叙事更完整；长度适配设计很实用
- Limitations: diffusion 路线整体仍较新；文本侧生态和工具链不如 AR 成熟

## 我的笔记 / 开放问题

- LLaDA-o 特别有意思的一点是，它没有把“统一”理解成一种单调的目标，而是承认统一系统内部也可以有不同 diffusion 子机制。
- 后续非常值得看的是：这种长度自适应是否会成为 omni 模型的通用组件，而不只局限于 diffusion 路线。

## 相关笔记

- [MMaDA 笔记](./mmada-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)

## 参考资料

- You et al., "LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model", arXiv, 2026. https://arxiv.org/abs/2603.01068
- Official repository. https://github.com/ML-GSAI/LLaDA-o
