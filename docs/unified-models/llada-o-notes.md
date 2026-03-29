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

LLaDA-o 的核心不是“又一个 diffusion 统一模型”，而是把统一对象明确拆成离散 masked diffusion 和连续 diffusion 两种子机制，再通过共享注意力主干和长度自适应策略把它们组织成一个可用的 omni model。

## 背景 / 问题设定

如果接受“统一模型也可以围绕 diffusion 建立”，接下来马上会遇到两个问题：

- 文本理解和视觉生成显然不适合同一种 diffusion 形式；
- 多模态输出长度差异极大，若没有长度适配机制，统一解码将非常低效。

LLaDA-o 正是围绕这两个问题设计的。它不是简单重复 MMaDA，而是在给 omni-diffusion 增加更细的内部结构。

## 记号

设：

- 文本序列为 \(x\)
- 图像连续变量为 \(y\)
- 共享注意力主干为 \(f_\theta\)
- 离散 masked diffusion 目标为 \(\mathcal{L}_{\text{disc}}\)
- 连续 diffusion 目标为 \(\mathcal{L}_{\text{cont}}\)
- 长度适配相关目标为 \(\mathcal{L}_{\text{len}}\)

## 核心思想

### 1. Mixture of Diffusion

LLaDA-o 不要求所有模态走完全相同的 diffusion 过程，而是把统一建立在一个 MoD 框架上：

- 文本理解更适合离散 masked diffusion；
- 视觉生成更适合连续 diffusion。

### 2. 共享主干而非共享所有细节

不同 diffusion 分支通过共享高效 attention backbone 耦合，因此统一仍然发生在上下文交互层，而不是变量形式层。

### 3. Length Adaptation 是核心工程机制

LLaDA-o 很重要的一点是，它把长度自适配从工程细节提升为方法设计的一部分。这说明作者意识到 unified model 的实际难点常常不是“能不能统一”，而是“统一后是否高效可用”。

## 一个简单示意图

```text
text input --------------------> discrete diffusion branch --+
                                                            |
image / visual target -------> continuous diffusion branch --+--> shared attention backbone --> outputs
                                                            |
                                                            +--> length-adaptive decoding control
```

## Architecture / Data Flow

LLaDA-o 的结构可以概括成“三明治式”：

1. 文本相关任务映射到离散 masked diffusion 分支；
2. 图像生成相关任务进入连续 diffusion 分支；
3. 两类分支通过共享 attention backbone 建立跨模态条件关系；
4. 最终通过 length adaptation 机制调整不同任务下的解码长度与计算预算。

这里最重要的结构判断是：统一发生在 backbone 和训练组织层，而不是通过粗暴要求“所有对象都服从同一 diffusion 公式”。

## Training Objective / Recipe

LLaDA-o 的训练可以理解为一个混合 diffusion 配方：

\[
\mathcal{L}
= \lambda_{\text{disc}} \mathcal{L}_{\text{disc}}
+ \lambda_{\text{cont}} \mathcal{L}_{\text{cont}}
+ \lambda_{\text{len}} \mathcal{L}_{\text{len}}.
\]

但更有信息量的是 recipe 的解释：

- 离散理解任务用于训练 masked diffusion 分支；
- 连续图像生成任务用于训练 continuous diffusion 分支；
- 混合多模态数据用于训练共享 backbone 的跨模态条件能力；
- 长度适配策略通过数据驱动方式学习不同任务下的更优推理长度。

论文更偏系统设计，而不是把这一切都细化成一套大推导，因此这里更适合作为基于文中信息的训练理解。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制是 Mixture of Diffusion 加 length adaptation。前者解决“统一但不一刀切”，后者解决“统一之后是否真能工作”。

### 它想解决统一模型里的哪种核心冲突？

它要解决的是统一 diffusion 模型内部的两层冲突：

- 文本与视觉需要不同 diffusion 形式；
- 多模态输出长度不一致会导致统一推理浪费算力。

### 它的关键设计为什么成立？

因为很多跨模态统一需求其实只要求共享上下文处理核心，而不要求所有模态的生成动力学完全同构。LLaDA-o 正是沿着这个判断，把差异放进 MoD 内部，再把统一放到 backbone 上。

### 它相比相邻方法最大的不同点是什么？

和 MMaDA 相比，它更强调内部机制拆分与长度效率；和 Show-o / Transfusion 相比，它把“混合建模”完全放进 diffusion 叙事而不是 AR + diffusion 二分法中。

## 与相邻方法的关系

### 它和谁最像？

最像 MMaDA，因为两者都站在 diffusion-centered unified model 这一阵营。

### 它和谁差异最大？

和 Emu3 差异最大。Emu3 把统一压缩到单一 next-token prediction；LLaDA-o 则认为 unified model 内部完全可以保留不同 diffusion 子机制。

### 它继承了什么？

它继承了 diffusion 在视觉生成中的强归纳偏置，也继承了 unified model 社区对共享 backbone 的偏好。

### 它修正了什么？

它修正的是 omni-diffusion 方法中“不同模态和长度差异被低估”的问题。

### 它留下了什么问题？

它留下的问题是：这种长度自适应是否会成为 unified model 的通用组件？以及 diffusion-centered 方法能否在文本交互体验上真正接近成熟 AR-LM。

## 重要细节

- Architecture: Mixture of Diffusion + shared attention backbone + length adaptation
- Objective: 文本离散 masked diffusion + 图像连续 diffusion + 长度适配训练
- Data: 多模态理解、图像生成和长度分布多样的混合任务数据
- Evaluation: multimodal understanding、text-to-image generation、efficiency-oriented omni benchmarks
- Strengths: 对 omni-diffusion 内部结构刻画更细；效率意识很强
- Limitations: diffusion 路线整体仍较新；文本侧工具链和用户体验尚不如 AR 模型成熟

## My Take / Why It Matters

LLaDA-o 在统一模型的发展链条里，更像是 omni-diffusion 方法走向工程可用化的一步。它最有价值的地方，是把“统一”从抽象概念推进到“不同 diffusion 子机制如何共存、计算如何分配”的层面。

它的局限则在于，这条路线的真正上限还没有被完全验证。但如果未来统一模型越来越重视效率与长度控制，LLaDA-o 这种思路很可能会被更广泛地借鉴。

## 相关笔记

- [MMaDA 笔记](./mmada-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)

## 参考资料

- You et al., "LLaDA-o: An Effective and Length-Adaptive Omni Diffusion Model", arXiv, 2026. https://arxiv.org/abs/2603.01068
- Official repository. https://github.com/ML-GSAI/LLaDA-o
