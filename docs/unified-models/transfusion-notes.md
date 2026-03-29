# Transfusion 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

## 一句话总结

Transfusion 的关键洞见是：统一模型不必把所有模态都压成同一种变量形式，文本适合 next-token prediction，图像更适合 diffusion，而真正值得共享的是跨模态上下文主干。

## 背景 / 问题设定

Transfusion 面向的是统一模型中的两种极端：

- 全部离散化、全部 AR；
- 语言模型和图像扩散模型彼此独立，只在接口层简单拼接。

论文认为前者牺牲了图像生成的归纳偏置，后者又丢失了统一上下文建模能力。于是问题变成：

- 能否共享一个 transformer 主干？
- 同时又让文本和图像保留最合适的输出动力学？

## 记号

设：

- 文本 token 序列为 \(x = (x_1,\dots,x_m)\)
- 图像连续表示或 latent 序列为 \(y = (y_1,\dots,y_n)\)
- 共享主干为 \(f_\theta\)
- 文本语言模型目标为 \(\mathcal{L}_{\text{LM}}\)
- 图像 diffusion 目标为 \(\mathcal{L}_{\text{Diff}}\)

## 核心思想

### 1. 同一主干处理离散与连续模态

Transfusion 的统一点不在“单一 token space”，而在“单一 transformer 主干”。文本和图像都通过共享主干组织条件关系。

### 2. 文本走 AR，图像走 diffusion

这是方法的核心。它承认不同模态的最佳生成偏置不同，因此让文本保留 next-token prediction，让图像保留 diffusion。

### 3. 统一主干，模态专属 I/O

论文进一步指出，模态专属的输入输出层不会破坏统一，反而能提高性能，因为它们负责的是模态接口适配，而不是跨模态认知本体。

## 一个简单示意图

```text
text tokens ------------------> text embedding -----------+
                                                         |
image / latent / patches --> image embedding / adapter --+--> shared transformer
                                                         |
                                                         +--> LM head --> next text token
                                                         |
                                                         +--> diffusion head --> image denoising step
```

## Architecture / Data Flow

Transfusion 的结构可以理解为三层：

1. 模态专属输入层，把文本和图像转换到共享主干可处理的表示空间；
2. 共享 transformer 主干，负责混合模态上下文建模；
3. 模态专属输出层，把共享隐状态分别映射到语言模型头和 diffusion 头。

这里真正重要的不是“有两个头”，而是共享主干始终看到混合模态上下文，因此文本和图像的条件依赖关系是在同一个认知空间里被建立的。

如果换一个更结构性的表述，它在做的是：

- 输入变量不统一；
- 输出动力学不统一；
- 但上下文语义空间统一。

这也是 Transfusion 和很多“统一 token 路线”最本质的差异。

## Training Objective / Recipe

训练目标可以写成双目标混合：

\[
\mathcal{L} = \lambda_{\text{LM}} \mathcal{L}_{\text{LM}} + \lambda_{\text{Diff}} \mathcal{L}_{\text{Diff}}.
\]

但更有信息量的是 recipe：

- 纯文本样本用于维持语言建模能力；
- 图文样本用于学习跨模态条件关系；
- 图像生成样本用于训练 diffusion 分支；
- 所有样本都通过共享主干更新跨模态上下文表征。

这意味着共享主干同时承受两类学习信号：

- 语言侧要求它保留因果 next-token 结构；
- 图像侧要求它提供足够强的条件表示，以支撑 diffusion denoising。

论文很强调 scaling behavior，这说明它真正关心的不只是“混合 recipe 能不能工作”，而是“它在规模增大后是否仍然稳定且优于极端路线”。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制是“共享主干 + 模态专属动力学”。统一发生在条件建模上，而不是变量形式上。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是：文本和图像是否必须共享同一种生成形式。Transfusion 的回答是“不必”，并把冲突从“是否统一”改写成“哪些部分统一、哪些部分专属”。

### 它的关键设计为什么成立？

因为文本和图像虽然输出变量不同，但共享大量条件依赖结构。共享 transformer 可以学习这种依赖，而输出头只需负责把共享隐状态投影回各自最自然的生成空间。

### 它相比相邻方法最大的不同点是什么？

和 Emu3 / Chameleon 相比，它不相信 AR 足以解释一切；和 Orthus 相比，它更明确地把视觉侧表述成标准 diffusion，而不是“AR backbone 下外挂连续头”的折中版本。

## 与相邻方法的关系

### 它和谁最像？

最像 Orthus 和 Show-o，因为三者都承认图像不必完全服从语言 AR 规律，只是处理方式不同。

### 它和谁差异最大？

和 Emu3 差异最大。Emu3 把统一推进到目标函数层面；Transfusion 则把统一停在主干层面。

### 它继承了什么？

它继承了 transformer 作为共享上下文引擎的思路，也继承了 diffusion 在图像生成上的强归纳偏置。

### 它修正了什么？

它修正的是“统一模型必须采用统一变量形式”这一假设，把统一重新定义为“共享条件建模核心”。

### 它留下了什么问题？

它留下的问题是：共享主干是否会在规模变大后被两类梯度持续撕扯？以及统一主干到底会更偏向语言推理还是更偏向视觉生成。

## 重要细节

- Architecture: 共享 transformer 主干 + 模态专属输入层 / 输出层
- Objective: 文本 next-token prediction + 图像 diffusion
- Data: 纯文本、图文混合、图像生成相关训练数据
- Evaluation: text-only、image generation、cross-modal benchmarks、scaling behavior
- Strengths: 不强迫统一变量形式；统一逻辑清晰；很重视规模化行为
- Limitations: 训练与推理比纯 AR 更复杂；视觉分支仍保留 diffusion 采样成本

## My Take / Why It Matters

Transfusion 在统一模型链条里很像一个“中间主义坐标点”。它最有价值的思想，是把统一从“所有模态做同一件事”改成“所有模态在同一认知核心里交互，但保留各自最合适的输出动力学”。

它的局限在于，这条路线天然比极端统一更复杂，也更容易被质疑“不够纯粹”。但正因为它足够现实，它对后来大量 hybrid unified model 都有很强启发意义。

## 相关笔记

- [Emu3 笔记](./emu3-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Orthus 笔记](./orthus-notes.md)

## 参考资料

- Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", arXiv, 2024. https://arxiv.org/abs/2408.11039
