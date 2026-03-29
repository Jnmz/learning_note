# Chameleon 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Chameleon: Mixed-Modal Early-Fusion Foundation Models

## 一句话总结

<<<<<<< HEAD
Chameleon 是“单一 token space + early fusion + 统一自回归”路线里的代表作，它把图像和文本尽早写进同一个 token 流，使统一模型可以在任意图文顺序下进行理解与生成。

## 背景 / 问题设定

Chameleon 面向的是多模态系统设计里一个非常根本的判断：

- 如果 transformer 本来就擅长处理长 token 序列
- 那是否应当尽量早地把不同模态都转进这个序列世界

这与晚融合或外挂式视觉模块形成鲜明对比。它要解决的问题不是“怎样加一个视觉接口”，而是“怎样把视觉真正纳入 foundation model 的主语言”。
=======
Chameleon 是统一 token 路线里的代表作之一，它通过 mixed-modal early fusion 把图像和文本统一成一个 token 流，使同一个 foundation model 能在任意图文顺序下进行理解和生成。

## 背景 / 问题设定

Chameleon 面向的核心问题是：如果语言模型已经能在一个长 token 序列上处理复杂上下文，那么能否直接把图像也写进这个序列里，从而避免晚融合或外挂式多模态设计？

这条路线的吸引力是：

- 架构简单
- 统一程度高
- 任意交错顺序的图文建模自然成立
>>>>>>> origin/main

## 记号

设：

- 文本 token 为 \(x = (x_1,\dots,x_m)\)
- 图像 token 为 \(y = (y_1,\dots,y_n)\)
- 混合序列为 \(s = (s_1,\dots,s_T)\)
<<<<<<< HEAD
- 共享 transformer 为 \(f_\theta\)
=======
- 模型参数为 \(\theta\)
>>>>>>> origin/main

## 核心思想

### 1. Early Fusion

<<<<<<< HEAD
Chameleon 不在高层才拼接视觉和语言，而是在输入早期就把图像和文本统一成一个 token 序列。

### 2. 任意顺序的 mixed-modal generation

因为训练对象就是混合序列，模型天然支持图文任意交错。这一点对长篇 multimodal document generation 尤其重要。

### 3. 把视觉生成纳入 foundation model 范式

Chameleon 不把图像生成视作外挂扩散器的职责，而是视作统一 token 模型内部自然可做的续写任务。

## Architecture / Data Flow

Chameleon 的结构非常纯粹：

```text
text ----------> text tokenizer ---------+
                                         |
image ---------> image tokenizer --------+--> mixed-modal token stream --> shared AR transformer --> next token
```

数据流的关键点有三个：

1. 图像必须先被离散化成适合语言模型处理的 token。
2. 图像 token 与文本 token 在 very early stage 就进入同一序列。
3. transformer 对整条混合序列做标准自回归建模，不再为视觉设置单独主干。

这种结构的优点是上下文特别统一：

- 图像条件文本生成是普通序列续写
- 文本条件图像生成也是普通序列续写
- 图文交错长文档则是更一般的序列续写

## Training Objective / Recipe

Chameleon 的训练目标本身很简单，就是统一混合序列上的 autoregressive 建模：

\[
\mathcal{L}_{\text{AR}}
= -\sum_{i=1}^{T}\log p_\theta(s_i \mid s_{<i}).
\]

真正难的部分在训练 recipe，而不是数学形式：

- 需要一个足够强的 image tokenizer 把视觉变成可建模 token
- 需要 carefully curated mixed-modal data 让模型见到多种图文顺序
- 需要稳定训练 recipe，防止语言能力或视觉能力一侧塌缩

这也是论文特别强调 stable training approach 的原因。对于 Chameleon 这样的系统，loss 很朴素，但训练组织才是成败关键。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

真正关键的机制是单一 token space 上的 mixed-modal early fusion。没有 early fusion，它就只是“带视觉插件的 LLM”；没有单一 token space，它也无法把图像生成自然写成序列续写。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是“多模态系统是否应该从一开始就被分成不同模块”这一冲突。Chameleon 的回答是：如果目标是 foundation model，那就应该尽量早地统一。

### 它的关键设计为什么成立？

因为一旦图像被成功离散化，很多原本看似不同的任务都能写成条件序列建模。换句话说，它的成立依赖于“视觉离散化足够成功”这个前提。

### 它相比相邻方法最大的不同点是什么？

和 Janus、Orthus、Transfusion 最大的不同是，它不太接受“模态专属路径”这件事。和 Emu3 的差异则更小，主要在于 Emu3 更进一步把视频也纳入统一叙事。

## 与相邻方法的关系

### 它和谁最像？

最像 Emu3。两者都高度信任 AR 和 token 统一，只是 Emu3 更进一步走向图像、视频、文本的全模态一致接口。

### 它和谁差异最大？

和 Janus 差异最大，因为 Janus 认为视觉入口不应过度统一，而 Chameleon 恰恰主张尽早统一。

### 它继承了什么？

它继承了语言模型在长序列上下文建模上的优势，并把这种优势直接扩展到视觉 token。

### 它修正了什么？

它修正的是“视觉必须走另一套模型”的旧范式，使 mixed-modal generation 变成 foundation model 内部能力。

### 它留下了什么问题？

它留下的问题主要集中在视觉 tokenizer：如果图像离散化不够强，统一 token 流就可能成为信息瓶颈，而不是能力桥梁。
=======
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
>>>>>>> origin/main

## 重要细节

- Architecture: mixed-modal early-fusion token-based transformer
<<<<<<< HEAD
- Objective: 统一混合序列上的 autoregressive 建模
- Data: 文本、图像、图文交错序列
- Evaluation: captioning、VQA、文本生成、图像生成、mixed-modal generation
- Strengths: 统一程度高；接口纯粹；长图文交错生成很自然
- Limitations: 强依赖 image tokenizer；图像生成质量和采样效率受 token 路线限制

## My Take / Why It Matters

Chameleon 在统一模型谱系里像一个极其清晰的基准点。它最有价值的思想，是证明“统一 token 流”不只是一个抽象愿景，而是可以落成 foundation model 的具体系统。

它的局限也同样清楚：它把很多难题都压到视觉离散化和长序列建模上了。但正因为它把路线走得足够纯粹，后续所有走混合路线的方法几乎都可以看成是在回答 Chameleon 留下的问题。
=======
- Objective: 统一 token 序列上的 autoregressive 建模
- Data: 文本、图像、混合图文序列
- Evaluation: captioning、VQA、text generation、image generation、mixed-modal generation
- Strengths: 统一程度极高；接口自然；图文任意交错很优雅
- Limitations: 强依赖视觉 tokenizer；图像生成效率与质量受离散化限制

## 我的笔记 / 开放问题

- Chameleon 是很多后续 unified model 的重要参照物，因为它把“统一 token 流”这件事做得非常彻底。
- 但它也把统一模型的老问题暴露得很明显：视觉离散化究竟是统一的桥梁，还是最终瓶颈？
>>>>>>> origin/main

## 相关笔记

- [Emu3 笔记](./emu3-notes.md)
- [Janus 笔记](./janus-notes.md)
- [Orthus 笔记](./orthus-notes.md)

## 参考资料

- Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models", arXiv, 2024. https://arxiv.org/abs/2405.09818
