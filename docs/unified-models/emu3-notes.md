# Emu3 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Emu3: Next-Token Prediction is All You Need
  - Emu3 official repository

## 一句话总结

Emu3 的核心立场非常鲜明：统一多模态模型不一定需要 diffusion、U-Net 或组合式架构，只要把文本、图像、视频都离散成 token，再坚持 next-token prediction，就有可能把理解和生成统一到一个纯自回归 transformer 中。

## 背景 / 问题设定

在 Emu3 之前，很多多模态系统依然沿着两条分裂路线发展：

- 理解模型通常是视觉编码器加 LLM
- 生成模型通常是 diffusion 或扩散式视频模型

Emu3 的问题意识很明确：如果 next-token prediction 已经在语言上如此成功，那么它能否成为统一文本、图像、视频的单一训练原则？

## 记号

设：

- 文本 token 序列为 \(t = (t_1,\dots,t_m)\)
- 图像或视频离散 token 序列为 \(v = (v_1,\dots,v_n)\)
- 混合多模态序列记为 \(s = (s_1,\dots,s_T)\)
- 模型参数记为 \(\theta\)

## 核心思想

### 1. 一切都变成 token

Emu3 的统一方式非常直接：把图像、视频、文本全部表示成离散 token，然后交给同一个 transformer 处理。这意味着模型不再区分“这是语言建模器”还是“这是视觉生成器”，而是统一成序列建模器。

### 2. 一切都做 next-token prediction

与 Show-o、Transfusion 这类混合目标路线不同，Emu3 刻意追求目标函数上的极简主义。文本、图像、视频都在同一套 autoregressive next-token prediction 框架下训练。

### 3. 用统一 token 叙事换取可扩展性

Emu3 认为，统一 token 表示与统一 AR 目标能让训练和推理变得更可扩展，尤其是在混合模态序列越来越长的情况下。

## 关键机制

### 离散视觉 tokenizer

Emu3 是否成立，很大程度上取决于视觉 tokenizer 的质量。只有当图像和视频能被压缩成兼顾语义和保真的离散 token 序列时，“纯 AR 万能论”才有机会成立。

### 单主干序列建模

模型使用单个 transformer 从头训练，输入可以是：

- text-only
- image-text
- video-text
- interleaved multimodal sequences

统一主干的好处是，理解和生成天然共享上下文处理能力。

### 统一采样接口

因为所有输出都是 token，Emu3 的文本生成、图像生成、视频生成都可视作同一采样过程的不同实例，只是目标 token 类型不同。

## 直觉 / 理解

Emu3 有一种“把多模态问题做窄、把统一原则做硬”的味道。它不试图为不同模态保留专属生成机制，而是押注一个结论：只要 token 化足够强，AR 就足够通用。

## 与其他方法的关系

### 对比 Show-o

Show-o 是文本 AR 加图像离散 diffusion；Emu3 则把文本、图像、视频全部压到 AR next-token prediction 上。Emu3 的统一程度更激进。

### 对比 Transfusion

Transfusion 认为文本和图像的最佳建模方式不同，因此采用 language modeling + diffusion 的混合。Emu3 则正相反，认为统一目标本身就值得坚持。

### 对比 Chameleon

两者都偏早融合、偏 token 统一，但 Emu3 把视频也更明确地纳入统一建模叙事，并更强调“从头训练、纯 AR recipe”。

## 重要细节

- Architecture: 单一 transformer 主干，从头训练
- Objective: 统一的 next-token prediction
- Data: 文本、图像、视频与混合多模态序列
- Evaluation: multimodal perception、text-to-image、text-to-video、跨模态生成
- Strengths: 训练目标非常统一；系统叙事简单；易于扩展到视频
- Limitations: 强依赖视觉 tokenizer；AR 图像 / 视频采样代价高；高分辨率下序列长度压力大

## 我的笔记 / 开放问题

- Emu3 的价值不只是效果，而是它对“统一模型是否必须保留模态专属目标”给出了非常强硬的反例。
- 但它也把问题转移到了 tokenizer 上：如果视觉离散化不够强，主干越统一，瓶颈反而越集中。

## 相关笔记

- [Chameleon 笔记](./chameleon-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)

## 参考资料

- Wang et al., "Emu3: Next-Token Prediction is All You Need", arXiv, 2024. https://arxiv.org/abs/2409.18869
- Official repository. https://github.com/baaivision/Emu3
