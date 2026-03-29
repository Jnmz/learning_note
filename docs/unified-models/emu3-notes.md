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

<<<<<<< HEAD
Emu3 是统一模型里最彻底的 AR 路线之一：它坚持把文本、图像、视频全部变成同一种序列接口，并用 next-token prediction 作为唯一核心训练原则。

## 背景 / 问题设定

Emu3 面向的是统一模型设计中的一个根本分歧：

- 一派认为不同模态应该保留不同建模形式，例如文本 AR、图像 diffusion
- 另一派认为只要统一 token 接口足够强，自回归序列建模就可以覆盖所有模态

Emu3 明显站在后一派。它真正想验证的不是“一个多模态模型能不能工作”，而是“next-token prediction 是否足以统一理解和生成”。
=======
Emu3 的核心立场非常鲜明：统一多模态模型不一定需要 diffusion、U-Net 或组合式架构，只要把文本、图像、视频都离散成 token，再坚持 next-token prediction，就有可能把理解和生成统一到一个纯自回归 transformer 中。

## 背景 / 问题设定

在 Emu3 之前，很多多模态系统依然沿着两条分裂路线发展：

- 理解模型通常是视觉编码器加 LLM
- 生成模型通常是 diffusion 或扩散式视频模型

Emu3 的问题意识很明确：如果 next-token prediction 已经在语言上如此成功，那么它能否成为统一文本、图像、视频的单一训练原则？
>>>>>>> origin/main

## 记号

设：

- 文本 token 序列为 \(t = (t_1,\dots,t_m)\)
- 图像或视频离散 token 序列为 \(v = (v_1,\dots,v_n)\)
- 混合多模态序列记为 \(s = (s_1,\dots,s_T)\)
<<<<<<< HEAD
- 视觉 tokenizer 记为 \(\tau_{\text{vis}}\)
=======
>>>>>>> origin/main
- 模型参数记为 \(\theta\)

## 核心思想

### 1. 一切都变成 token

<<<<<<< HEAD
Emu3 的统一方式非常直接：先把图像、视频、文本全部变成同一种离散序列接口，再让同一个 transformer 处理这些 token。

### 2. 一切都做 next-token prediction

与 Show-o、Transfusion、LLaDA-o 这种混合目标路线不同，Emu3 刻意追求目标函数极简主义。所有模态统一成 AR next-token prediction。

### 3. 统一接口优先于模态专门化

Emu3 的方法判断是：如果系统接口足够统一，模型就能在一个共享上下文空间里自然学习“看图说话”“看视频回答”“根据文本生成图像 / 视频”等不同任务。

## Architecture / Data Flow

Emu3 的结构可以概括为“视觉 tokenizer + 统一 token 序列 + 单一 AR transformer”。

数据流大致如下：

```text
text --------------------------+
                               |
image/video --> visual tokenizer +--> unified token sequence --> AR transformer --> next token
```

具体来说：

1. 文本直接使用语言 tokenizer。
2. 图像和视频先经过视觉 tokenizer \(\tau_{\text{vis}}\)，变成离散视觉 token。
3. 不同模态 token 用特殊边界符和任务模板组织进同一序列。
4. 单个 transformer 在该序列上做因果建模。
5. 输出时继续预测下一个 token，直到得到文本、图像 token 或视频 token。

这一结构最关键的地方不在主干本身，而在于“接口彻底共享”：

- 输入接口共享
- 上下文缓存共享
- 训练目标共享
- 采样机制共享

因此 Emu3 的统一不是“一个主干配多个头”，而更接近“一个序列世界模型”。

## Training Objective / Recipe

Emu3 的训练目标极其干净：所有任务都被改写成 next-token prediction。

\[
\mathcal{L}_{\text{AR}}
= - \sum_{i=1}^{T} \log p_\theta(s_i \mid s_{<i}).
\]

但真正重要的是不同数据如何对齐到这个目标：

- 理解样本：`[image/video tokens] + [question text] -> [answer text]`
- 生成样本：`[prompt text] -> [image tokens]` 或 `-> [video tokens]`
- 交错样本：`[text][image][text][video]...` 的长序列续写

从 recipe 角度看，它更像一个大规模混合数据训练系统：

- 需要高质量视觉 tokenizer 先把图像 / 视频离散化
- 需要多种任务模板统一成序列预测
- 需要同一主干在长混合序列上保持稳定训练

论文更偏系统 recipe，而不是复杂数学拆解。就训练理解而言，它最大的特点不是 loss 新，而是把几乎所有多模态任务都压进了同一 loss。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制是统一 token / sequence interface。Emu3 不是因为 transformer 有多新，而是因为它把所有模态都转进同一个序列世界里。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是“统一模型是否必须保留模态专属生成机制”这一冲突。Emu3 的回答是：不一定，至少在接口层和目标层，可以极端统一。

### 它的关键设计为什么成立？

成立的前提有两个：

1. 视觉 tokenizer 必须足够强，既压缩又保真；
2. AR transformer 必须能吸收多模态长序列里的局部与全局依赖。

换句话说，Emu3 的成功更多建立在“好的视觉离散化 + 强大的序列建模器”这一组合上。

### 它相比相邻方法最大的不同点是什么？

最大的不同点是它几乎不为图像和视频保留特殊建模地位。Show-o、Transfusion、Orthus、LLaDA-o 都承认视觉侧可能需要不同动力学；Emu3 则把这一步彻底压平。

## 与相邻方法的关系

### 它和谁最像？

最像 Chameleon。两者都非常相信 token 统一和 AR 序列建模，只是 Emu3 更明确地把视频也纳入统一路线。

### 它和谁差异最大？

和 MMaDA、LLaDA-o 这种 omni-diffusion 路线差异最大，因为后者认为统一也可以围绕 diffusion 展开。

### 它继承了什么？

它继承了 Chameleon 这类 mixed-token 模型的早融合和统一接口思想，同时继承了语言模型在长序列因果建模上的成功经验。

### 它修正了什么？

它修正的是“视觉必须外挂专用生成器”的思路，把图像和视频更彻底地纳入单主干世界观。

### 它留下了什么问题？

它留下的核心问题是：视觉 tokenizer 会不会成为整个系统的单点瓶颈？以及高分辨率图像 / 长视频是否会让 AR 采样代价变得过高。

## 重要细节

- Architecture: 视觉 tokenizer + unified sequence interface + 单一 AR transformer
- Objective: 统一的 next-token prediction
- Data: 文本、图像、视频与长交错多模态序列
- Evaluation: multimodal perception、text-to-image、text-to-video、跨模态续写
- Strengths: 结构和目标极度统一；系统叙事简单；天然支持图文视频交错上下文
- Limitations: 极度依赖视觉离散化质量；图像 / 视频 AR 采样成本高；统一代价被集中到 tokenizer 和长序列建模

## My Take / Why It Matters

Emu3 在统一模型发展链条里的意义非常大，因为它把“统一接口优先”这条路线推进得几乎没有退路。它最有价值的思想，是用一个足够强硬的反例迫使社区重新思考：模态专属目标究竟是必须的，还是历史包袱。

它的局限也同样明显。Emu3 越成功，越说明视觉 tokenizer 是统一模型里的真正“隐形主干”；而一旦 tokenizer 不够强，所有统一优势都会反过来变成集中瓶颈。这恰恰也是它对后续方法最重要的启发。
=======
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
>>>>>>> origin/main

## 相关笔记

- [Chameleon 笔记](./chameleon-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)

## 参考资料

- Wang et al., "Emu3: Next-Token Prediction is All You Need", arXiv, 2024. https://arxiv.org/abs/2409.18869
- Official repository. https://github.com/baaivision/Emu3
