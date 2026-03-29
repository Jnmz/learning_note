# Orthus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads

## 一句话总结

Orthus 的独特点在于：它坚持自回归统一主干，但不再把图像硬压成离散 token，而是让共享 transformer 输出分别路由到 language head 和 diffusion head，以同时处理离散文本和连续图像特征。

## 背景 / 问题设定

Orthus 所针对的问题和 Transfusion 有相似之处：纯 token 化视觉会带来信息损失，而纯组合式系统又损失统一性。论文试图探索第三条路线：

- 主干仍然保持自回归
- 文本仍然是离散 token
- 图像则改成连续特征并由 diffusion head 生成

## 记号

设：

- 文本 token 为 \(x\)
- 图像连续特征为 \(z\)
- 共享主干为 \(f_\theta\)
- 语言头为 \(h_{\text{LM}}\)
- 图像 diffusion 头为 \(h_{\text{diff}}\)

## 核心思想

### 1. Fully AR Backbone

Orthus 认为自回归建模对于跨模态相关性的表达非常直接，因此继续坚持 fully AR formulation。

### 2. 连续视觉信号而非离散 VQ token

与很多 unified AR 模型不同，Orthus 不想在图像侧承受 VQ 离散化带来的信息损失，因此改用更柔性的连续视觉特征。

### 3. 模态专属 head 负责最终输出

共享 transformer 负责理解上下文关系，而真正的模态差异留给输出头去处理：文本由 LM head 预测，图像由 diffusion head 生成。

## 关键机制

### Soft Alternative to VQ

论文强调，它是通过替换已有统一 AR 模型中的 VQ 操作，引入一种更 soft 的视觉表示方式，再加一个 diffusion head 来构建 Orthus。这个设计很实用，因为它降低了从已有统一模型出发迁移到新架构的成本。

### Interleaved Image-Text Generation

Orthus 不仅做 text-to-image 和 image understanding，也强调长篇图文交错生成。这使它更像一个“多模态内容写作器”而不仅是问答器或画图器。

### Efficient Construction

作者强调训练构建成本相对友好，这一点对统一模型很重要，因为很多架构看起来漂亮，但复现与扩展门槛极高。

## 直觉 / 理解

Orthus 有点像 AR 统一路线对自身局限的一次修补：既保留 AR 的统一叙事，又承认图像不一定适合被粗暴离散化，因此把视觉复杂性后移到 diffusion head。

## 与其他方法的关系

### 对比 Chameleon / Emu3

Chameleon 和 Emu3 更接近“图像也变 token”；Orthus 则认为视觉连续性值得保留。

### 对比 Transfusion

两者都采用共享主干加连续视觉输出，但 Orthus 更强调 fully AR backbone 与 modality-specific heads 这一组合。

### 对比 Show-o

Show-o 在图像侧采用离散 mask prediction；Orthus 则用 diffusion head 直接生成连续视觉特征。

## 重要细节

- Architecture: AR transformer backbone + LM head + diffusion head
- Objective: 文本 AR 预测 + 图像 diffusion 生成
- Data: 图像理解数据、图像生成数据、图文交错数据
- Evaluation: text-to-image、VQA、mixed-modality generation
- Strengths: 保留 AR 统一性同时降低视觉离散化损失；对交错内容生成友好
- Limitations: 仍需 diffusion 头与连续视觉管线；系统接口比纯 token 模型更复杂

## 我的笔记 / 开放问题

- Orthus 说明“是否统一”与“是否离散化”并不是同一个问题。可以统一主干，但不统一输出变量形式。
- 一个后续值得关注的问题是：在更大规模下，AR backbone + diffusion head 会更接近 Transfusion，还是会走向更强的视觉专门化？

## 相关笔记

- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)

## 参考资料

- Kou et al., "Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads", arXiv, 2024. https://arxiv.org/abs/2412.00127
