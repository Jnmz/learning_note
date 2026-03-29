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

Emu3 是统一模型里最纯粹的 AR 路线之一：它坚持把文本、图像、视频都变成同一种序列接口，并用 next-token prediction 作为统一理解与生成的核心训练原则。

## 背景 / 问题设定

Emu3 面向的是统一模型设计中的一个根本分歧：

- 一派认为不同模态应保留不同建模形式，例如文本 AR、图像 diffusion；
- 另一派认为只要统一 token 接口足够强，自回归序列建模就能覆盖所有模态。

Emu3 明显站在后一派。它想验证的不是“一个多模态模型能不能工作”，而是更强的问题：next-token prediction 是否足以成为统一文本、图像、视频的单一训练范式。

## 记号

设：

- 文本 token 序列为 \(t = (t_1,\dots,t_m)\)
- 图像或视频离散 token 序列为 \(v = (v_1,\dots,v_n)\)
- 混合多模态序列为 \(s = (s_1,\dots,s_T)\)
- 视觉 tokenizer 为 \(\tau_{\text{vis}}\)
- 统一自回归模型为 \(f_\theta\)

## 核心思想

### 1. 一切都变成 token

Emu3 的统一方式非常直接：先把图像、视频、文本全部转换成离散 token，再让同一个 transformer 处理这些 token。

### 2. 一切都做 next-token prediction

与 Show-o、Transfusion、LLaDA-o 这类混合目标路线不同，Emu3 刻意追求目标函数极简主义。不同模态统一成 AR next-token prediction。

### 3. 接口统一优先于模态专门化

Emu3 的方法判断是：如果接口统一得足够彻底，那么同一个主干就能自然吸收跨模态条件关系，而不必在结构上为每种模态预留独立动力学。

## 一个简单示意图

```text
text --------------------------+
                               |
image/video --> visual tokenizer +--> unified token sequence --> AR transformer --> next token
```

## Architecture / Data Flow

Emu3 的系统结构可以概括为“视觉 tokenizer + 混合 token 序列 + 单一 AR transformer”。

具体的数据流如下：

1. 文本直接经语言 tokenizer 转为文本 token；
2. 图像和视频通过视觉 tokenizer \(\tau_{\text{vis}}\) 转为离散视觉 token；
3. 文本 token 与视觉 token 通过特殊边界符和任务模板组织进统一序列 \(s\)；
4. 统一 transformer 在整个序列上做因果建模；
5. 输出仍然是“下一个 token”，只不过这个 token 可能属于文本、图像或视频。

因此 Emu3 的统一不是“共享一个大 backbone，加很多头”，而是更接近“所有模态都进入同一个序列世界”。

这一点带来两个直接结果：

- 理解任务可以被看作视觉条件下的文本续写；
- 生成任务可以被看作文本条件下的视觉 token 续写；
- 视频任务则只是更长、更强结构化的视觉 token 序列续写。

## Training Objective / Recipe

Emu3 的训练目标极其干净：

\[
\mathcal{L}_{\text{AR}}
= - \sum_{i=1}^{T} \log p_\theta(s_i \mid s_{<i}).
\]

但真正重要的是不同数据怎样被组织成这个目标：

- 理解样本：`[image/video tokens] + [question text] -> [answer text]`
- 图像生成样本：`[prompt text] -> [image tokens]`
- 视频生成样本：`[prompt text] -> [video tokens]`
- 交错样本：`[text][image][text][video]...` 的长序列续写

所以 recipe 的难点不在 loss，而在三件事：

1. 视觉 tokenizer 必须足够强；
2. 多任务模板必须能被统一成稳定的序列接口；
3. 单一主干必须能在长混合序列上维持训练稳定。

这也是为什么 Emu3 的方法贡献更像“统一 recipe 的完整性”，而不是某个单独新模块。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制是 unified token / sequence interface。Emu3 的真正主张不是“transformer 很强”，而是“只要序列接口统一，理解和生成都可以被还原成同一种学习问题”。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是“统一模型是否必须保留模态专属目标”这一冲突。Emu3 的回答是：未必，只要离散化和序列化足够强。

### 它的关键设计为什么成立？

因为一旦视觉 tokenizer 足够好，很多原本看似不同的任务都会退化为条件序列续写问题。此时 AR transformer 的通用性就被最大化复用。

### 它相比相邻方法最大的不同点是什么？

和 Show-o、Transfusion、LLaDA-o 最大的不同点是：Emu3 不给图像和视频保留特殊地位。它不是“统一主干、分化动力学”，而是“连动力学也尽量统一”。

## 与相邻方法的关系

### 它和谁最像？

最像 Chameleon。二者都高度信任 token 统一和 AR 建模，只是 Emu3 更明确地把视频也纳入统一叙事。

### 它和谁差异最大？

和 MMaDA、LLaDA-o 这类 diffusion-centered unified model 差异最大，因为后者认为统一也可以围绕扩散动力学展开。

### 它继承了什么？

它继承了 Chameleon 这种早融合、统一 token 流的思想，也继承了语言模型在长序列上下文建模上的成功经验。

### 它修正了什么？

它修正的是“视觉生成必须外挂专门生成器”的看法，把图像和视频更彻底地纳入统一主干。

### 它留下了什么问题？

它留下的核心问题是：视觉 tokenizer 会不会成为整个系统的单点瓶颈？以及高分辨率图像和长视频是否会让 AR 采样代价高到不具现实性。

## 重要细节

- Architecture: 视觉 tokenizer + unified token sequence + 单一 AR transformer
- Objective: 统一的 next-token prediction
- Data: 文本、图像、视频与交错多模态长序列
- Evaluation: multimodal perception、text-to-image、text-to-video、跨模态续写
- Strengths: 结构与目标极度统一；系统叙事简单；天然支持长交错上下文
- Limitations: 严重依赖视觉离散化质量；图像 / 视频 AR 采样成本高；统一代价被集中到 tokenizer 和长序列建模

## My Take / Why It Matters

Emu3 在统一模型发展链条里的意义非常大，因为它把“统一接口优先”推进得几乎没有退路。它最有价值的思想，是用一个足够强硬的 AR 反例迫使社区重新思考：模态专属目标究竟是必要归纳偏置，还是历史惯性。

它的局限同样明显。Emu3 越成功，越说明视觉 tokenizer 是 unified model 里的真正“隐形主干”；一旦 tokenizer 不够强，所有统一优势都会反过来变成集中瓶颈。

## 相关笔记

- [Chameleon 笔记](./chameleon-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)

## 参考资料

- Wang et al., "Emu3: Next-Token Prediction is All You Need", arXiv, 2024. https://arxiv.org/abs/2409.18869
- Official repository. https://github.com/baaivision/Emu3
