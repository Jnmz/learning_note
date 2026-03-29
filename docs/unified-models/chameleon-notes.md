# Chameleon 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Chameleon: Mixed-Modal Early-Fusion Foundation Models

## 一句话总结

Chameleon 是统一 token 路线里的代表作之一，它通过 mixed-modal early fusion 把图像和文本统一成一个 token 流，使同一个 foundation model 能在任意图文顺序下进行理解和生成。

## 背景 / 问题设定

Chameleon 面向的核心问题是：如果语言模型已经能在一个长 token 序列上处理复杂上下文，那么能否直接把图像也写进这个序列里，从而避免晚融合或外挂式多模态设计？

这条路线的吸引力是：

- 架构简单
- 统一程度高
- 任意交错顺序的图文建模自然成立

## 记号

设：

- 文本 token 为 \(x = (x_1,\dots,x_m)\)
- 图像 token 为 \(y = (y_1,\dots,y_n)\)
- 混合序列为 \(s = (s_1,\dots,s_T)\)
- 模型参数为 \(\theta\)

## 核心思想

### 1. Early Fusion

Chameleon 不是在高层才拼接视觉和语言，而是尽早把它们都转进同一 token 流，让 transformer 从一开始就在混合序列上建模。

### 2. 任意顺序 mixed-modal generation

因为训练对象是统一 token 序列，所以图像和文本可以任意交错出现。模型既能看图说话，也能根据上下文在一长段文本中插入图像。

### 3. Token-Based Foundation Model

Chameleon 把“视觉生成”也纳入 foundation model 叙事，而不是单独训练一个图像扩散器。这一点对后续很多 unified model 都有启发作用。

## 关键机制

### 图像离散 token 化

Chameleon 的成立强依赖视觉 tokenization。图像必须被转成适合语言模型处理的 token 序列，才能真正进入 early-fusion 架构。

### 稳定训练 recipe

论文特别强调 stable training approach，这说明统一 token 路线虽然概念简单，但训练上并不天然稳定，需要专门的 recipe 才能同时维持语言和视觉能力。

### Long-Form Mixed-Modal Generation

Chameleon 不是只做图文单轮任务，而是明确展示了长程 mixed-modal generation 的潜力，这也是它的重要示范意义。

## 直觉 / 理解

Chameleon 很像在说：“如果 transformer 能处理任何 token 序列，那就别太早给模态设边界。” 这是统一模型里最纯粹的一种美学。

## 与其他方法的关系

### 对比 Emu3

两者都高度信任 token 统一和 AR 建模，但 Emu3 进一步把视频也纳入同一叙事，并更强调“next-token prediction 足够解释一切”。

### 对比 Janus

Janus 认为视觉编码不该被过度统一；Chameleon 则更接近“尽早统一、尽量共享”。

### 对比 Orthus / Transfusion

Orthus 和 Transfusion 更愿意保留连续视觉建模；Chameleon 则代表离散 token 统一的极致路线。

## 重要细节

- Architecture: mixed-modal early-fusion token-based transformer
- Objective: 统一 token 序列上的 autoregressive 建模
- Data: 文本、图像、混合图文序列
- Evaluation: captioning、VQA、text generation、image generation、mixed-modal generation
- Strengths: 统一程度极高；接口自然；图文任意交错很优雅
- Limitations: 强依赖视觉 tokenizer；图像生成效率与质量受离散化限制

## 我的笔记 / 开放问题

- Chameleon 是很多后续 unified model 的重要参照物，因为它把“统一 token 流”这件事做得非常彻底。
- 但它也把统一模型的老问题暴露得很明显：视觉离散化究竟是统一的桥梁，还是最终瓶颈？

## 相关笔记

- [Emu3 笔记](./emu3-notes.md)
- [Janus 笔记](./janus-notes.md)
- [Orthus 笔记](./orthus-notes.md)

## 参考资料

- Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models", arXiv, 2024. https://arxiv.org/abs/2405.09818
