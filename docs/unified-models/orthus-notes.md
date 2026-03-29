# Orthus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads

## 一句话总结

Orthus 可以看成 AR 统一路线的一次内部修正：它保留共享自回归主干来统一跨模态上下文，但不再强迫图像经由硬离散 VQ token，而是把视觉复杂性后移到模态专属 diffusion head。

## 背景 / 问题设定

Orthus 瞄准的是纯 AR 统一路线中的一个现实问题：

- 如果把图像全压成离散 token，视觉细节与保真会被压缩；
- 如果回到“语言模型 + 独立生成器”，又会失去统一上下文接口。

因此它真正要解决的是：如何在保留 AR 主干统一性的同时，缓和视觉离散化带来的信息损失。

## 记号

设：

- 文本 token 为 \(x\)
- 图像连续视觉特征为 \(z\)
- 共享 AR 主干为 \(f_\theta\)
- 语言头为 \(h_{\text{LM}}\)
- 图像 diffusion 头为 \(h_{\text{diff}}\)
- 文本目标为 \(\mathcal{L}_{\text{text}}\)
- 图像目标为 \(\mathcal{L}_{\text{img}}\)

## 核心思想

### 1. Fully AR backbone 保留不变

Orthus 仍然认为 AR 主干对跨模态上下文依赖关系的表达很自然，因此没有放弃统一 AR 核心。

### 2. 图像表示从硬离散走向更柔性的连续形式

它不再把视觉彻底压成 VQ token，而是保留更适合生成的连续视觉特征。

### 3. 用 modality-specific heads 处理最终差异

共享主干负责跨模态建模，语言头负责文本 token，diffusion 头负责视觉合成。也就是说，Orthus 的统一策略是“共享理解核心，专属输出动力学”。

## 一个简单示意图

```text
text tokens ----------------------------+
                                        |
image / prompt --> visual representation +--> shared AR transformer
                                        |
                                        +--> LM head ---------> text tokens
                                        |
                                        +--> diffusion head --> image features / image
```

## Architecture / Data Flow

Orthus 的信息流可以理解为：

1. 文本仍然进入统一 AR 主干，保留标准因果序列建模接口；
2. 图像或视觉目标先通过更柔性的连续表示进入系统，而不是硬离散 VQ token；
3. 共享主干负责组织图文交错上下文；
4. 主干输出分别路由到 LM head 和 diffusion head；
5. 文本侧继续做 token prediction，视觉侧则由 diffusion head 完成细节生成。

这说明 Orthus 不是简单“给 AR 模型加一个图像头”，而是在做职责重新分配：

- AR 主干负责“内容结构与跨模态因果关系”；
- diffusion head 负责“视觉细节实现”。

## Training Objective / Recipe

Orthus 的训练可以理解为混合两类样本：

- 理解 / 文本任务：图像条件文本输出或纯文本续写；
- 图像生成任务：文本或交错上下文条件下的视觉生成。

对应目标写成：

\[
\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \lambda_{\text{img}} \mathcal{L}_{\text{img}}.
\]

但真正有信息量的是配方：

- 共享 AR 主干统一处理交错上下文；
- 文本头在标准 LM 目标下训练；
- 视觉头在 diffusion / 连续特征生成目标下训练；
- 图文交错数据帮助主干学习跨模态续写与长程上下文组织。

论文里一个值得注意的点是，它强调自己可以看成对已有 unified AR 模型中 VQ 部分的 soft 替代。这意味着 Orthus 的 recipe 其实是在 AR 路线内部做增量修正，而不是推翻整条路线。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

关键机制是“AR backbone + modality-specific heads”。这一步把统一和视觉保真两个看似冲突的目标拆成了不同模块职责。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是 AR 统一路线里的视觉信息瓶颈。为了统一而硬做 VQ 离散化，会让图像质量成为整个系统的短板。

### 它的关键设计为什么成立？

因为 unified model 的很多收益来自共享上下文结构，而不一定来自共享最终输出变量。只要 AR 主干继续组织跨模态因果关系，视觉头就可以相对独立地承担细节生成。

### 它相比相邻方法最大的不同点是什么？

和 Chameleon / Emu3 相比，Orthus 更明确地拒绝“全离散 token 统一”；和 Transfusion 相比，它对 AR backbone 的保留更执着，像是在 AR 路线内部修补，而不是另起一条新混合路线。

## 与相邻方法的关系

### 它和谁最像？

最像 Transfusion，因为两者都采用共享主干加模态专属输出路线。

### 它和谁差异最大？

和 Chameleon、Emu3 差异最大，因为后两者更接近“图像也只是 token”，而 Orthus 明确认为视觉连续性不能轻易抹掉。

### 它继承了什么？

它继承了 AR 主干统一上下文的思想，也继承了交错图文生成作为统一界面的目标。

### 它修正了什么？

它修正的是早期 AR unified model 对 VQ tokenization 依赖过重的问题。

### 它留下了什么问题？

它留下的问题是：共享 AR backbone 和 diffusion head 的组合，在更大规模下会不会逐渐收敛成类似 Transfusion 的更一般混合范式？以及它是否真正优于更纯粹的纯 AR 或纯 diffusion 路线。

## 重要细节

- Architecture: AR transformer backbone + LM head + diffusion head
- Objective: 文本 AR 目标 + 图像连续生成目标
- Data: 图像理解、图像生成、图文交错长序列数据
- Evaluation: text-to-image、VQA、interleaved image-text generation
- Strengths: 保住 AR 统一接口；减少硬离散带来的视觉损失；对长交错内容生成友好
- Limitations: 已不再是“纯 AR 一招鲜”；训练与推理链条比纯 token 路线更复杂

## My Take / Why It Matters

Orthus 在统一模型演化链条里的位置，很像 AR 统一路线的一次内生修正。它最有价值的思想，是提醒我们：统一不应该等同于“所有模态都被迫服从同一种表示”。

它的主要局限是，这条路线仍带有较强的工程折中色彩，理论上不如极端统一路线整齐。但正因为它更现实，它为后续很多“共享主干、专属输出”的 unified model 提供了很自然的中间模板。

## 相关笔记

- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)

## 参考资料

- Kou et al., "Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads", arXiv, 2024. https://arxiv.org/abs/2412.00127
