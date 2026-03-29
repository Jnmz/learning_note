# Orthus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads

## 一句话总结

<<<<<<< HEAD
Orthus 可以看成 AR 统一路线的一次有意识修正：它保留共享自回归主干来统一跨模态上下文，但不再强迫图像经由硬离散 VQ token，而是把视觉复杂性后移到模态专属 diffusion head。

## 背景 / 问题设定

Orthus 瞄准的是纯 AR 统一路线中的一个现实问题：

- 如果把图像全压成离散 token，会损失细节与保真
- 如果回到“语言模型 + 独立生成器”，又会失去统一上下文接口

因此它要解决的是：如何在保留 AR 主干统一性的同时，缓和视觉离散化带来的信息损失。
=======
Orthus 的独特点在于：它坚持自回归统一主干，但不再把图像硬压成离散 token，而是让共享 transformer 输出分别路由到 language head 和 diffusion head，以同时处理离散文本和连续图像特征。

## 背景 / 问题设定

Orthus 所针对的问题和 Transfusion 有相似之处：纯 token 化视觉会带来信息损失，而纯组合式系统又损失统一性。论文试图探索第三条路线：

- 主干仍然保持自回归
- 文本仍然是离散 token
- 图像则改成连续特征并由 diffusion head 生成
>>>>>>> origin/main

## 记号

设：

- 文本 token 为 \(x\)
<<<<<<< HEAD
- 图像连续视觉特征为 \(z\)
- 共享 AR 主干为 \(f_\theta\)
- 语言头为 \(h_{\text{LM}}\)
- 图像 diffusion 头为 \(h_{\text{diff}}\)
- 文本目标为 \(\mathcal{L}_{\text{text}}\)
- 图像目标为 \(\mathcal{L}_{\text{img}}\)

## 核心思想

### 1. Fully AR backbone 不变

Orthus 仍然认为 AR 主干对跨模态上下文依赖关系的表达很自然，因此没有放弃统一 AR 核心。

### 2. 图像表示从硬离散走向更柔性的连续形式

它不再把视觉彻底压成 VQ token，而是保留更适合生成的连续视觉特征。

### 3. 用 modality-specific heads 处理最终差异

共享主干负责跨模态建模，语言头负责文本 token，diffusion 头负责视觉合成。也就是说，Orthus 的统一策略是“共享理解核心，专属输出动力学”。

## Architecture / Data Flow

Orthus 的信息流可以理解为：

```text
text tokens ----------------------------+
                                        |
image / prompt --> visual representation +--> shared AR transformer
                                        |
                                        +--> LM head ---------> text tokens
                                        |
                                        +--> diffusion head --> image features / image
```

它和 Transfusion 看起来相似，但重点不同：

- Transfusion 更强调“文本 AR + 图像 diffusion”的统一主干组合
- Orthus 更强调“AR backbone 本身继续保留”，图像复杂性主要由 modality-specific heads 吸收

换句话说，Orthus 不是要弱化 AR，而是要避免 AR 路线被 VQ tokenization 拖住。

## Training Objective / Recipe

Orthus 的训练可以理解为混合两类样本：

- 理解 / 文本任务：图像条件文本输出或纯文本续写
- 图像生成任务：文本或交错上下文条件下的视觉生成

对应目标写成

\[
\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \lambda_{\text{img}} \mathcal{L}_{\text{img}}.
\]

但真正重要的不是式子，而是配方：

- 共享 AR 主干统一处理交错上下文
- 文本头在标准 LM 目标下训练
- 视觉头在 diffusion / 连续特征生成目标下训练
- 图文交错数据帮助主干学习跨模态续写和长程上下文组织

因此 Orthus 的 recipe 更像在说：AR 主干负责“思维链和内容结构”，扩散头负责“视觉细节实现”。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

关键机制是“AR backbone + modality-specific heads”。这一步把统一和视觉保真这两个看似冲突的目标拆成了不同模块职责。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是 AR 统一路线里的视觉信息瓶颈。也就是：如果为了统一而硬做 VQ 离散化，视觉质量可能成为瓶颈。

### 它的关键设计为什么成立？

因为很多统一收益其实来自共享上下文结构，而不一定来自共享最终输出变量。只要 AR 主干继续组织跨模态因果关系，视觉头就可以相对独立地承担细节生成。

### 它相比相邻方法最大的不同点是什么？

和 Chameleon / Emu3 相比，Orthus 更明确地拒绝“全离散 token 统一”；和 Transfusion 相比，它对 AR backbone 的保留更执着，像是在 AR 路线内部做补丁而不是改信仰。

## 与相邻方法的关系

### 它和谁最像？

最像 Transfusion，因为两者都采用共享主干加模态专属输出路线。

### 它和谁差异最大？

和 Chameleon、Emu3 差异最大，因为后两者更接近“图像也只是 token”，而 Orthus 明确认为视觉连续性不能轻易抹掉。

### 它继承了什么？

它继承了 AR 主干统一上下文的思想，也继承了图文交错生成作为统一接口的目标。

### 它修正了什么？

它修正的是早期 AR unified model 过度依赖 VQ tokenization 的问题。

### 它留下了什么问题？

它留下的问题是：共享 AR backbone 和 diffusion head 的组合，在更大规模下到底会不会收敛到类似 Transfusion 的混合范式？以及它是否真的比纯 diffusion / 纯 AR 更优，而不是仅仅更折中。
=======
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
>>>>>>> origin/main

## 重要细节

- Architecture: AR transformer backbone + LM head + diffusion head
<<<<<<< HEAD
- Objective: 文本 AR 目标 + 图像连续生成目标
- Data: 图像理解、图像生成、图文交错长序列数据
- Evaluation: text-to-image、VQA、interleaved image-text generation
- Strengths: 保住 AR 统一接口；减少硬离散带来的视觉损失；对长交错内容生成友好
- Limitations: 系统已不再是“纯 AR 一招鲜”；训练与推理链条比纯 token 路线更复杂

## My Take / Why It Matters

Orthus 在统一模型演化链条里的位置，很像 AR 统一路线的一次内生修正。它最有价值的思想是提醒我们：统一不应该等同于“所有模态都被迫服从同一种表示”。

它的主要局限是，这条路线仍然有较强工程折中色彩，理论上不如极端统一路线整齐。但正因为它更现实，它为后续很多“共享主干、专属输出”的统一方案提供了很自然的中间模板。
=======
- Objective: 文本 AR 预测 + 图像 diffusion 生成
- Data: 图像理解数据、图像生成数据、图文交错数据
- Evaluation: text-to-image、VQA、mixed-modality generation
- Strengths: 保留 AR 统一性同时降低视觉离散化损失；对交错内容生成友好
- Limitations: 仍需 diffusion 头与连续视觉管线；系统接口比纯 token 模型更复杂

## 我的笔记 / 开放问题

- Orthus 说明“是否统一”与“是否离散化”并不是同一个问题。可以统一主干，但不统一输出变量形式。
- 一个后续值得关注的问题是：在更大规模下，AR backbone + diffusion head 会更接近 Transfusion，还是会走向更强的视觉专门化？
>>>>>>> origin/main

## 相关笔记

- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)

## 参考资料

- Kou et al., "Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads", arXiv, 2024. https://arxiv.org/abs/2412.00127
