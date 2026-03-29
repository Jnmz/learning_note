# Janus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
  - Janus official repository

## 一句话总结

Janus 的核心贡献在于指出：统一多模态模型真正的冲突不一定来自“主干是否共享”，而更可能来自“理解和生成是否被迫共用同一种视觉编码”，因此它选择在共享 transformer 主干的同时，解耦视觉编码路径。

## 背景 / 问题设定

很多统一模型在设计上会默认“一个视觉编码器服务所有任务”，但 Janus 认为这一步本身就可能有问题：

- 理解需要高层语义、判别性特征
- 生成需要更适合重建和细节表达的视觉表示

如果二者共用完全相同的视觉编码路径，理解和生成就可能彼此妥协。

## 记号

设：

- 理解侧图像表示为 \(z_u\)
- 生成侧图像表示为 \(z_g\)
- 共享 transformer 主干记为 \(f_\theta\)
- 文本序列为 \(x\)

## 核心思想

### 1. 共享主干，不共享视觉编码

Janus 不是完全拆分成两个模型，而是在共享 transformer backbone 的前提下，把理解编码器和生成编码器解耦。

### 2. 视觉冲突主要出现在编码层

论文的核心判断是：理解与生成的矛盾，很多时候不是语言主干不能共享，而是视觉表示要求不同。于是将冲突局部化到视觉侧，是更高性价比的统一路线。

### 3. 用自回归框架统一任务接口

Janus 仍然偏向 autoregressive 统一叙事，通过统一 transformer 来处理跨模态上下文，只是在输入侧给理解和生成不同的视觉表征。

## 关键机制

### Decoupled Visual Encoding

理解路径使用更适合感知和语义抽象的视觉输入表示，生成路径则使用更适合视觉重建和图像生成的表示。这样，模型既保留了统一主干，又避免了单视觉编码器带来的信息粒度冲突。

### Unified Transformer Processing

虽然编码器解耦，但后续跨模态建模仍然在同一个 transformer 中完成，因此 Janus 不是“两个模型拼起来”，而是“两个视觉入口接一个共享认知核心”。

### 扩展性

这种设计的一个好处是，理解和生成两条路径理论上都能独立替换更强组件，而不需要重做整个统一架构。

## 直觉 / 理解

Janus 的价值在于它重新定义了“统一”。它并不执着于从输入到输出每一步都共享，而是更关心哪里应该共享、哪里应该解耦。这个判断比“统一得越彻底越好”更现实。

## 与其他方法的关系

### 对比 Chameleon / Emu3

Chameleon 和 Emu3 更倾向于把视觉和文本都压进统一 token 流。Janus 则认为，视觉表示层面的差异不应被过度抹平。

### 对比 Show-o

Show-o 把冲突处理为“文本 AR、图像离散 diffusion”；Janus 则把冲突处理为“理解视觉编码器和生成视觉编码器解耦”。

### 对比 DreamLLM

DreamLLM 更强调理解-创作协同；Janus 更强调理解-创作冲突的工程化拆解。

## 重要细节

- Architecture: 理解视觉编码器 + 生成视觉编码器 + 共享 transformer 主干
- Objective: 统一多模态 understanding / generation 训练
- Data: 多模态理解数据与图像生成数据
- Evaluation: VQA、captioning、text-to-image、统一多模态 benchmark
- Strengths: 把冲突局部化；兼顾统一与专门化；工程上更灵活
- Limitations: 形式上不如“全共享”统一；两条视觉路径会增加系统复杂度

## 我的笔记 / 开放问题

- Janus 很像统一模型里的“模块化现实主义”。它承认不同任务对视觉表示的需求并不相同。
- 一个值得继续跟踪的问题是：未来更强的 unified model，到底会更接近 Emu3 这种极致统一，还是更接近 Janus 这种共享核心、前端解耦？

## 相关笔记

- [DreamLLM 笔记](./dreamllm-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)

## 参考资料

- Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation", arXiv, 2024. https://arxiv.org/abs/2410.13848
- Official repository. https://github.com/deepseek-ai/Janus
