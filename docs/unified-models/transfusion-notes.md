# Transfusion 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

## 一句话总结

Transfusion 的核心思想是承认“文本和图像并不一定适合同一种建模形式”，因此在同一个 transformer 里，把文本交给 next-token prediction，把图像交给 diffusion，从而在统一主干中同时处理离散与连续模态。

## 背景 / 问题设定

统一模型常见有两种极端：

- 要么把所有模态都离散化，再统一做 AR
- 要么让系统由独立语言模型和独立扩散模型组合

Transfusion 认为这两条路线都不理想。纯 AR 图像建模在采样效率和高保真生成上吃亏，而“语言模型 + 扩散模型”组合又缺乏真正统一的上下文处理。

## 记号

设：

- 文本 token 序列为 \(x = (x_1,\dots,x_m)\)
- 图像 patch / latent 序列为 \(y = (y_1,\dots,y_n)\)
- 共享主干为 \(f_\theta\)
- 文本 loss 为 \(\mathcal{L}_{\text{LM}}\)
- 图像 diffusion loss 为 \(\mathcal{L}_{\text{Diff}}\)

## 核心思想

### 1. 同一主干同时处理离散与连续数据

Transfusion 的“统一”不建立在单一 token 空间，而建立在单一 transformer backbone 上。文本作为离散 token 进入，图像则通过连续表示进入。

### 2. 文本做 AR，图像做 diffusion

这是论文最关键的设计点。它不是试图把图像强行变成文本，也不是把文本强行变成扩散变量，而是允许不同模态保留最合适的生成机制。

### 3. 用模态专属输入输出层降低冲突

论文还指出，可以通过 modality-specific encoding / decoding layers 进一步提升性能。这意味着共享主干不代表输入输出层必须完全统一。

## 关键机制

### Mixed-Modality Sequences

Transfusion 把文本和图像组织进统一的混合模态序列里，让同一个 transformer 可以跨模态处理条件依赖关系。

### Dual Objective

训练目标可以概括为：

- 对文本部分做标准 language modeling
- 对图像部分做 diffusion denoising / score-like 学习

这使它同时继承了 LLM 和扩散模型各自擅长的部分。

### Scaling Laws

论文很强调 scaling 行为，这是它区别于很多“只提方法不提扩展规律”的统一模型之一。它试图说明这种混合 recipe 不只是能工作，而且有规模化潜力。

## 直觉 / 理解

Transfusion 很像统一模型里的“中间主义”。它既不完全相信“全部 token 化 + 纯 AR”，也不接受“直接拼两个大模型”。它的答案是：统一主干，共享上下文，模态专属动力学。

## 与其他方法的关系

### 对比 Emu3 / Chameleon

Emu3 和 Chameleon 更接近把图像也纳入 token-based AR 体系。Transfusion 则认为图像生成在连续扩散形式上更自然。

### 对比 Show-o

Show-o 使用离散图像 token 上的 mask prediction；Transfusion 则更明确地保留连续 diffusion 视角。

### 对比 Orthus

Orthus 是 AR 主干加模态专属 head；Transfusion 也是统一主干配合模态差异化输出，但它的连续视觉建模更接近标准 diffusion。

## 重要细节

- Architecture: 共享 transformer 主干 + 模态专属编码 / 解码层
- Objective: 文本 next-token prediction + 图像 diffusion
- Data: 文本和图像混合训练语料
- Evaluation: text-only、image generation、cross-modal benchmarks
- Strengths: 不强迫所有模态共用单一建模形式；扩展规律分析较完整
- Limitations: 系统比纯 AR 更复杂；连续视觉分支仍然需要 diffusion 风格采样

## 我的笔记 / 开放问题

- Transfusion 代表了一种很值得重视的立场：真正统一的不一定是“单一输出形式”，而可能是“单一上下文处理核心”。
- 未来这类方法的关键问题是，统一主干是否真能同时兼顾语言推理与高保真视觉扩散，而不在某一侧严重让步。

## 相关笔记

- [Emu3 笔记](./emu3-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Orthus 笔记](./orthus-notes.md)

## 参考资料

- Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", arXiv, 2024. https://arxiv.org/abs/2408.11039
