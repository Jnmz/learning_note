# Orthus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads

## 一句话总结

Orthus 可以看成 AR 统一路线的一次内部修正：它保留共享自回归主干来统一跨模态上下文，但不再强迫图像经由硬离散 VQ token，而是把视觉复杂性后移到模态专属 diffusion head，用“AR backbone + continuous visual features + modality-specific heads”去缓和纯 token 统一带来的视觉信息瓶颈。

## 背景 / 问题设定

Orthus 瞄准的是纯 AR 统一路线中的一个现实问题。

如果沿着 Chameleon、Emu3 一类路线一直走下去，我们会得到一个非常整齐的结论：

- 文本是 token；
- 图像也压成 token；
- 所有内容都做 autoregressive next-token prediction。

这条路线在概念上很漂亮，但它会遭遇一个具体问题：图像一旦被硬离散化，视觉细节、局部结构和保真度常常会被 tokenizer 或 codebook 上限卡住。于是纯 AR 路线的统一，可能会以视觉质量为代价。

另一方面，如果直接退回“语言模型 + 独立生成器”的系统拼接，又会失去统一上下文接口带来的优势：

- 图文交错长上下文不再天然统一；
- 理解和生成变成两个系统之间的接口协作；
- 主干难以真正共享跨模态因果结构。

因此 Orthus 真正要解决的问题是：

- 如何在保留 AR 主干统一性的同时，缓和视觉离散化带来的信息损失？
- 如果文本和图像不再共享完全同一种输出变量，那么 unified model 的“统一”还剩下什么？

Orthus 的回答很有代表性：

\[
\text{keep the AR backbone},
\qquad
\text{relax the image representation and output dynamics}.
\]

也就是说，它不是退出 AR 路线，而是在 AR 路线内部做一次结构修补。

## 记号

设：

- 文本 token 序列为 \(x = (x_1,\dots,x_m)\)
- 图像连续视觉特征或 latent 表示为 \(z\)
- 扩散中的噪声时刻为 \(t \in [0,1]\)
- 加噪后的视觉表示为 \(z_t\)
- 共享 AR 主干为 \(f_\theta\)
- 语言头为 \(h_{\text{LM}}\)
- 图像 diffusion 头为 \(h_{\text{diff}}\)
- 文本目标为 \(\mathcal{L}_{\text{text}}\)
- 图像目标为 \(\mathcal{L}_{\text{img}}\)

Orthus 的高层结构可以抽象成

\[
h = f_\theta(\text{text context}, \text{visual context}),
\]

随后根据输出模态不同，主干输出再分别流向

\[
h_{\text{LM}}(h)
\quad \text{or} \quad
h_{\text{diff}}(h, t).
\]

## 核心思想

### 1. Fully AR backbone 保留不变

Orthus 仍然认为 AR 主干对跨模态上下文依赖关系的表达很自然，尤其适合图文交错长序列里的因果组织。因此它没有放弃统一 AR 核心。

换句话说，它并不认为问题出在 AR backbone 本身，而更可能出在视觉变量被压得太硬。

### 2. 图像表示从硬离散走向更柔性的连续形式

它不再把视觉彻底压成 VQ token，而是保留更适合生成的连续视觉特征。这个选择非常关键，因为它意味着 Orthus 重新分配了统一系统中的难度：

- 不再强求图像完全服从离散 token 语言；
- 而是让视觉保留连续性，把细节恢复交给 diffusion head。

### 3. 用 modality-specific heads 处理最终差异

共享主干负责跨模态建模，语言头负责文本 token，diffusion 头负责视觉合成。也就是说，Orthus 的统一策略是：

\[
\text{shared contextual reasoning}
\; + \;
\text{modality-specific realization}.
\]

这一步把统一和视觉保真这两个看似冲突的目标拆成了不同模块职责。

## 一个简单示意图

```text
text tokens ------------------------------+
                                          |
image / prompt --> continuous visual path +--> shared AR transformer
                                          |
                                          +--> LM head ---------> text tokens
                                          |
                                          +--> diffusion head --> image features / image
```

## 详细推导

### 推导 1：Orthus 保留的是 AR 条件分解，而不是强制所有输出都变成离散 token

在纯 AR 统一路线里，我们常常假设所有模态输出都可以写成统一 token 序列 \(s\)，然后做

\[
p_\theta(s) = \prod_i p_\theta(s_i \mid s_{<i}).
\]

Orthus 的关键修正是：它仍然保留文本侧的 AR 分解，但不再要求图像最终输出也必须以同样的离散 token 形式完成。

对文本任务，条件语言模型形式仍然成立。若多模态上下文记为 \(c\)，则

\[
p_\theta(x \mid c)
=
\prod_{i=1}^{m} p_\theta(x_i \mid x_{<i}, c),
\]

对应损失为

\[
\mathcal{L}_{\text{text}}
=
- \sum_{i=1}^{m} \log p_\theta(x_i \mid x_{<i}, c).
\]

这说明 Orthus 不是放弃 AR，而是只把 AR 保留在它最自然、最有优势的地方：文本与跨模态因果上下文建模。

### 推导 2：图像侧改为连续视觉空间中的 diffusion 学习

与硬离散 token 路线不同，Orthus 在图像侧保留连续视觉表示 \(z\)，并用 diffusion head 学习生成。

用标准连续扩散记号，可以写出一条噪声路径。例如设真实视觉表示为 \(z_1\)，噪声为 \(z_0\)，则可写成

\[
z_t = \alpha_t z_1 + \sigma_t z_0.
\]

或者用更接近 flow matching 的线性路径写成

\[
z_t = (1-t)z_0 + t z_1.
\]

若 diffusion head 预测噪声或等价的去噪目标，则一个典型损失可写成

\[
\mathcal{L}_{\text{img}}
=
\mathbb{E}_{z_1,z_0,t}
\left[
\left\|
\epsilon_\theta(z_t, t, c) - z_0
\right\|_2^2
\right].
\]

其中 \(c\) 来自共享 AR 主干提供的条件上下文。

这一步的意义在于：Orthus 不再逼迫图像生成走“离散类别预测”这条链，而是承认视觉细节恢复更适合连续生成动力学。

### 推导 3：联合目标体现的是“主干统一，输出动力学分化”

把两边写在一起，Orthus 的总目标可以概括为

\[
\mathcal{L}
=
\lambda_{\text{text}} \mathcal{L}_{\text{text}}
+
\lambda_{\text{img}} \mathcal{L}_{\text{img}}.
\]

表面上看这是普通多任务加权，但真正关键的是参数依赖结构：

\[
\mathcal{L}_{\text{text}} = \mathcal{L}_{\text{text}}(f_\theta, h_{\text{LM}}),
\qquad
\mathcal{L}_{\text{img}} = \mathcal{L}_{\text{img}}(f_\theta, h_{\text{diff}}).
\]

这里共享的是 \(f_\theta\)，分化的是最终输出头。也就是说，Orthus 并不认为“所有模态最后都得做同一种预测”才算统一；它认为更值得共享的是：

\[
\text{cross-modal context organization}.
\]

而模态最终的实现方式可以不同。

### 推导 4：为什么这可以看成对 VQ 路线的 soft 替代

若纯 AR 图像路线采用离散 VQ token \(u\)，则图像建模近似写成

\[
p_\theta(u \mid c)
=
\prod_{j=1}^{n} p_\theta(u_j \mid u_{<j}, c).
\]

这条路线的问题在于：视觉质量高度依赖 \(u\) 的离散化质量。一旦 VQ tokenizer 丢掉细节，后续 AR 主干再强也只能在受损空间里续写。

Orthus 改成连续视觉表示 \(z\) 后，等价于把“图像表示压缩为少量离散类别”的假设放宽成“图像表示保留连续可生成结构”。于是它相当于做了：

\[
\text{hard discrete image interface}
\rightarrow
\text{soft continuous image interface}.
\]

这也是为什么可以把它理解成对统一 AR 路线中 VQ 部分的一次 soft 替代，而不是整条路线的推翻。

## 架构理解

### 1. 为什么 Orthus 仍然属于 AR 统一路线

虽然它在图像侧用了 diffusion head，但 Orthus 仍然深深属于 AR 统一路线，因为：

- 跨模态上下文仍由 AR transformer 主干组织；
- 图文交错内容仍在因果序列框架下被建模；
- 文本接口和长上下文续写能力仍然以 AR 为核心。

因此它不是“转投 diffusion 阵营”，而是“在 AR 阵营内部做结构修正”。

### 2. 为什么说它重新分配了职责

Orthus 的核心不是“多了一个图像头”，而是职责划分发生了变化：

- AR 主干负责内容结构、跨模态因果关系和长上下文组织；
- diffusion head 负责视觉细节实现和保真恢复。

这种划分本质上是在说：

\[
\text{structure} \neq \text{realization}.
\]

文本和图像可以共享结构层，但不必共享最终实现层。

### 3. 为什么它比 Transfusion 更像 AR 路线内部修补

Transfusion 也是共享主干 + 模态专属输出，但它更像从一开始就站在“文本离散、图像连续”的混合建模立场上。Orthus 则带有更强的 AR 统一背景，它更像是在说：

- 我仍然相信 AR 主干；
- 我只是觉得 VQ 离散化把视觉压坏了；
- 所以把视觉细节交给 diffusion head。

这使它在气质上更像 AR 统一路线内部的一次内生修补。

## 训练流程

根据论文和公开材料，Orthus 的训练重点不在发明复杂损失，而在于如何在保留 AR 主干统一性的同时，把连续视觉生成能力稳定接上去。

### 阶段 1：用共享 AR 主干学习交错图文上下文

主干仍然要从文本和图文交错数据中学习长上下文组织、图文依赖和标准因果建模能力。这一步保住的是 unified AR interface。

### 阶段 2：用 diffusion head 学会视觉细节恢复

随后图像生成相关数据驱动 diffusion head 学习如何在共享语义条件下恢复连续视觉内容。这一步保住的是视觉保真与生成质量。

### 阶段 3：通过后训练强化 interleaved behavior

Orthus 特别强调 interleaved image-text generation，这意味着在后训练阶段还需要进一步强化：

- 图文长上下文切换；
- 共享主干与 diffusion head 的协同；
- 在 mixed-modal document 中同时维持语义连贯和视觉生成能力。

## 直觉 / 理解

我对 Orthus 的理解是：它像纯 AR unified model 路线的一次很自然的“自我修正”。如果一路坚持“所有模态都做 token”，最终很容易碰到视觉保真瓶颈；但如果直接放弃统一主干，又会丢掉 AR 路线最有价值的部分。Orthus 的折中很明确：

- 继续相信 AR 主干在上下文建模上的价值；
- 不再坚持图像必须以硬离散 token 形式出现；
- 把视觉复杂性留给专门的连续生成头。

这让它成为一种很有代表性的中间模板：不够“纯粹”，但很现实。

## 与相邻方法的关系

### 对比 Chameleon / Emu3

Chameleon、Emu3 更相信“图像也只是 token”，Orthus 则明确认为视觉连续性不能轻易抹掉。它们的根本差异不在主干，而在图像表示是否必须硬离散化。

### 对比 Transfusion

Transfusion 与 Orthus 最像，因为两者都采用共享主干加模态专属输出路线。不同在于，Transfusion 更像从一开始就接受 hybrid route，而 Orthus 更像是从 AR unified 路线内部修出一条连续视觉出口。

### 对比 Show-o / Show-o2

Show-o 系列也在统一主干里承认视觉不必完全服从文本 AR 规律。Orthus 与它们相似之处在于都把图像生成交给专门动力学；差异在于 Orthus 更直接围绕“AR backbone 的保留”来组织整个方法。

### 对比 Janus

Janus 把冲突定位到视觉入口编码，选择入口解耦；Orthus 则更像把冲突定位到图像表示与输出动力学，选择在输出端和视觉生成路径上修正。

## 重要细节

- Architecture: AR transformer backbone + LM head + diffusion head
- Objective: 文本 AR 目标 + 图像连续生成目标
- Representation: 文本离散 token 与视觉连续特征共存，图像不再强依赖硬离散 VQ token
- Data: 图像理解、图像生成、图文交错长序列数据
- Evaluation: text-to-image、VQA、interleaved image-text generation
- Strengths: 保住 AR 统一接口；减少硬离散带来的视觉损失；对长交错内容生成友好
- Limitations: 已不再是“纯 AR 一招鲜”；训练与推理链条比纯 token 路线更复杂；统一形式上没有极端路线整齐

## 我的笔记 / 开放问题

### 我的笔记

我觉得 Orthus 的价值，在于它没有把 unified model 的讨论卡死在“要么全 AR、要么完全放弃统一”这两个选项里。它说明 AR 路线也可以自我修正：不一定非要让图像像文本一样工作，但仍然可以保住一个共享的 AR 认知核心。

这使它在 unified model 谱系里很像一个“现实主义中间站”。很多后来方法哪怕不完全长成 Orthus，也都在沿着类似思路问：哪些东西必须统一，哪些东西不必硬统一？

### 开放问题

- 共享 AR backbone 和 diffusion head 的组合，在更大规模下会不会逐渐演化成类似 Transfusion 的更一般混合范式？
- 如果视觉 tokenizer 或连续表示进一步改进，Orthus 这种 soft 替代路线会不会比硬离散 token 路线更明显占优？
- 主干共享带来的跨模态收益，是否足以抵消 diffusion 分支增加的训练和推理复杂度？
- 当视频也纳入时，这种“AR 主干 + 连续视觉头”是否还能保持同样的简洁性？

## 相关笔记

- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)
- [Janus 笔记](./janus-notes.md)

## 参考资料

- Kou et al., "Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads", arXiv, 2024. https://arxiv.org/abs/2412.00127
- Hugging Face paper page. https://huggingface.co/papers/2412.00127

