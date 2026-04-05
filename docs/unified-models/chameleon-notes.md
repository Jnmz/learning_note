# Chameleon 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - Chameleon: Mixed-Modal Early-Fusion Foundation Models

## 一句话总结

Chameleon 是“单一 token space + early fusion + 统一自回归”路线里的代表作：它把文本和图像尽早写进同一个 token 流，用同一个 autoregressive transformer 从零训练理解与生成能力，从而把 mixed-modal document 建模推进成 foundation model 级别的问题，而不是外挂式多模态系统问题。

## 背景 / 问题设定

Chameleon 面向的是多模态系统设计里的一个非常根本的判断：

1. 如果 transformer 本来就擅长处理长 token 序列，那是否应尽量早地把不同模态都转进这个序列世界？
2. 如果目标是一个真正的 multimodal foundation model，那么图像是否还应该依赖独立生成器或晚融合视觉模块？
3. 若图像和文本都进入同一 token 流，模型是否可以自然支持任意图文顺序下的理解与生成？

这与典型的晚融合或外挂式视觉模块形成鲜明对比。很多多模态系统的做法是：

- 先用视觉 encoder 提取特征，
- 再在中后段把视觉信息送给语言模型，
- 真正的图像生成则交给另一套扩散或生成系统。

Chameleon 明确反对这种分层思路。它的主张是：如果我们真的想要统一模型，就不该只统一高层语义接口，而应尽可能从输入开始统一。

因此它要解决的问题不是“怎样给 LLM 加一个视觉接口”，而是：

\[
\text{how to make vision native to the token stream of a foundation model}.
\]

这条路线之所以有吸引力，是因为它承诺了三件事：

- 架构极简；
- 接口极统一；
- 图文任意交错的长程建模天然成立。

## 记号

设：

- 文本 token 序列为 \(x = (x_1,\dots,x_m)\)
- 图像输入为 \(I\)
- 图像经视觉 tokenizer 后得到的离散视觉 token 为 \(y = (y_1,\dots,y_n)\)
- 混合多模态序列为 \(s = (s_1,\dots,s_T)\)
- 视觉 tokenizer 为 \(\tau_{\text{img}}\)
- 共享 AR transformer 为 \(f_\theta\)
- 统一训练损失为 \(\mathcal{L}_{\text{AR}}\)

Chameleon 的输入统一可以抽象写成

\[
y = \tau_{\text{img}}(I),
\qquad
s = \operatorname{concat}(x, y, \text{special tokens}).
\]

随后模型在统一序列 \(s\) 上进行标准因果建模：

\[
p_\theta(s) = \prod_{i=1}^{T} p_\theta(s_i \mid s_{<i}).
\]

## 核心思想

### 1. Early Fusion 不是细节，而是方法立场

Chameleon 的第一个核心点是 early fusion。它不在高层才拼接视觉和语言，而是在输入早期就把图像和文本写入同一个 token 序列。

这意味着统一不是发生在“主干中途读到视觉条件”，而是从一开始就发生在：

\[
\text{text tokens} \cup \text{image tokens}
\rightarrow
\text{one sequence}.
\]

因此 Chameleon 并不是“带视觉插件的 LLM”，而是试图让视觉从第一步起就成为主序列语言的一部分。

### 2. Mixed-modal generation 是序列建模的自然结果

Chameleon 很重要的一点，是它不把 mixed-modal generation 当作一个额外外挂能力。因为训练对象本来就是混合序列，所以：

- 图像条件文本生成是序列续写，
- 文本条件图像生成也是序列续写，
- 任意图文交错文档生成仍然只是序列续写。

于是它的一个核心判断可以写成：

\[
\text{multimodal generation}
\subset
\text{mixed-sequence continuation}.
\]

### 3. 统一 foundation model 不一定要保留模态专属路径

Chameleon 与 Janus、Orthus、Transfusion 这类路线的根本分歧在于：它基本不接受“模态专属主路径”这件事。它更相信的是：

\[
\text{one tokenizer family}
+
\text{one sequence interface}
+
\text{one AR backbone}
\Rightarrow
\text{one foundation model}.
\]

这条路线很激进，因为它把很多系统复杂性直接消掉了；但也正因此，它把困难全部前移到了视觉离散化和长序列稳定训练上。

## 一个简单示意图

```text
text ----------> text tokenizer ---------+
                                         |
image ---------> image tokenizer --------+--> mixed-modal token stream --> shared AR transformer --> next token
```

## 详细推导

### 推导 1：混合模态建模本质上仍是统一的自回归分解

对 Chameleon 来说，一旦文本 token 和图像 token 被组织进同一个混合序列 \(s\)，所有训练就退化为标准语言模型形式：

\[
p_\theta(s) = \prod_{i=1}^{T} p_\theta(s_i \mid s_{<i}).
\]

于是负对数似然为

\[
\mathcal{L}_{\text{AR}}
=
- \sum_{i=1}^{T} \log p_\theta(s_i \mid s_{<i}).
\]

这个公式的关键不在于新，而在于它对 unified model 的含义非常强：Chameleon 明确拒绝为图像单独设计另一套生成目标。

更具体地看：

若做 image-conditioned text generation，可把序列组织成

\[
s = [y,\, x^{\text{prompt}},\, x^{\text{answer}}],
\]

于是模型实际拟合的是

\[
p_\theta(x^{\text{answer}} \mid y, x^{\text{prompt}})
=
\prod_{j=1}^{M}
p_\theta(x^{\text{answer}}_j \mid y, x^{\text{prompt}}, x^{\text{answer}}_{<j}).
\]

若做 text-to-image，则可组织成

\[
s = [x,\, y],
\]

于是模型拟合的是

\[
p_\theta(y \mid x)
=
\prod_{j=1}^{n} p_\theta(y_j \mid x, y_{<j}).
\]

因此在 Chameleon 眼里，理解和图像生成并不是两类 fundamentally different tasks，而只是同一个 AR factorization 在不同数据模板下的实例。

### 推导 2：Early Fusion 的实质是让跨模态依赖在最前缀层面被学习

假设有一类 late-fusion 系统，其高层表示写成

\[
h = F_{\text{late}}(x, E_{\text{img}}(I)).
\]

这里视觉信息先通过单独的图像编码器 \(E_{\text{img}}\) 被压成特征，再在较后层与文本融合。

Chameleon 则选择先把图像转成视觉 token

\[
y = \tau_{\text{img}}(I),
\]

再与文本一起形成序列 \(s\)，直接进入主干：

\[
h = f_\theta([x;y]).
\]

这两种设计的差异不只是“融合早一点还是晚一点”，而是依赖结构本身不同。

在 late fusion 下，模型主要学习的是

\[
p_\theta(\text{text output} \mid x, E_{\text{img}}(I)).
\]

而在 Chameleon 的 early fusion 下，模型学习的是

\[
p_\theta(s_i \mid s_{<i}),
\]

其中前缀 \(s_{<i}\) 可以同时包含文本和图像 token。这意味着跨模态依赖从序列最前缀开始就以统一形式存在，而不是先各走各路、后面再对齐。

所以 Early Fusion 在 Chameleon 里真正重要的不是“早”，而是：

\[
\text{cross-modal dependency}
\rightarrow
\text{native prefix dependency}.
\]

### 推导 3：图像生成被还原成视觉 token 续写问题

Chameleon 最激进的一点，是把图像生成也纳入同一个 AR backbone，而不是调用外部扩散器。设文本条件为 \(x\)，图像被离散化为 token 序列 \(y=(y_1,\dots,y_n)\)，那么图像生成目标直接写成

\[
p_\theta(y \mid x)
=
\prod_{j=1}^{n} p_\theta(y_j \mid x, y_{<j}).
\]

这看似和普通语言建模没有区别，但它隐含了一个很强的假设：

\[
\text{image} \approx \text{visual sentence over a discrete vocabulary}.
\]

只有当视觉 tokenizer 足够强时，这个假设才成立。否则：

- token 序列无法保留足够视觉细节；
- AR 模型必须在一个劣质离散空间里逐 token 生成；
- 图像质量和采样效率都会受损。

所以 Chameleon 的生成能力并不是“AR 自然万能”，而是“AR 在足够好的视觉离散化上也可以工作”。

### 推导 4：统一损失的简洁把训练难点转移到 tokenizer 与稳定性

Chameleon 的总损失只有一项：

\[
\mathcal{L}_{\text{AR}}
=
- \sum_{i=1}^{T}\log p_\theta(s_i \mid s_{<i}).
\]

这意味着方法表面非常干净，但实际困难被整体重分配了。粗略地说，总难度可以理解为

\[
\text{difficulty}
\approx
\text{tokenization quality}
+
\text{mixed-modal data quality}
+
\text{training stability}
+
\text{AR sampling cost}.
\]

Chameleon 论文特别强调 stable training approach，其实正说明：当我们把所有模态都强行纳入同一个 AR token 流后，loss 虽然统一了，但训练并不会自动稳定。

这也是它比“简单把图像 token 塞进 LLM”更难的地方。真正难的不是公式，而是如何让这个统一公式在大规模 mixed-modal 数据上可训练、可扩展。

## 架构理解

### 1. Chameleon 的统一发生在输入界面最前端

很多 unified model 虽然也共享主干，但视觉往往先被编码成另一种内部对象，之后再送入主干。Chameleon 的不同在于：

- 图像不先变成“外部条件特征”；
- 而是尽早变成与文本并列的 token；
- 共享主干看到的从一开始就是混合 token 流。

所以它的统一比很多多模态 LLM 更彻底，因为统一发生在 token interface，而不只是 hidden-state interface。

### 2. 为什么它天然支持任意图文顺序

一旦训练对象是 mixed-modal sequence，那么

\[
[text][image][text][image]\dots
\]

这样的结构本来就是标准训练样本。于是任意图文顺序下的理解与生成，不需要通过额外模式切换来实现，而是被统一序列建模自然吸收。

这也是 Chameleon 对 multimodal document generation 的重要意义：它把长文档级混合模态结构直接变成 foundation model 的原生数据格式。

### 3. 为什么它从零训练而不是简单在现有 LLM 上外挂

Chameleon 的一个重要现实是，这么彻底的 early fusion 很难仅靠“在现有 LLM 上加一点适配器”得到。因为：

- 视觉 token 会改变词表统计结构；
- mixed-modal prefix 会改变注意力依赖模式；
- 图像 token 生成会引入和文本不同的局部模式与长度分布。

这意味着，若想真正得到 mixed-modal early-fusion foundation model，往往需要从更底层开始联合训练，而不是简单拼接现有能力。

## 训练流程

根据论文与公开材料，Chameleon 的训练重点不在复杂损失，而在稳定地组织大规模 mixed-modal 数据和统一词表生态。

### 阶段 1：建立文本 token 与视觉 token 的共同序列统计

模型首先要学会新加入的视觉 token 在统一序列中意味着什么。这一步不只是“识别新 token”，而是学习：

- 文本与图像 token 的共存分布；
- 图文边界和插入模式；
- 视觉 token 的局部依赖结构。

### 阶段 2：用 mixed-modal data 学习图文任意顺序的续写

当模型初步适应视觉 token 后，再用多种图文组合样本训练，让它真正学会：

- \(p(\text{text} \mid \text{image, text prefix})\)
- \(p(\text{image} \mid \text{text prefix})\)
- \(p(\text{mixed sequence continuation} \mid \text{mixed prefix})\)

这一步是 Chameleon 区别于普通 MLLM 的关键，因为它真正训练的是 mixed-modal continuation，而不是单向理解。

### 阶段 3：稳定训练与行为对齐

当所有模态都进入同一 AR 流后，训练很容易出现语言能力、视觉生成能力、mixed-modal 行为之间的相互拉扯。因此 Chameleon 特别强调稳定训练。这里的重点不是发明新损失，而是保证：

- 文本能力不会被视觉 token 破坏；
- 图像生成不会因序列建模过弱而失稳；
- mixed-modal 长上下文行为能保持一致。

## 直觉 / 理解

我对 Chameleon 的理解是：它像 unified model 领域里一个极其纯的“基准点”。很多工作会说自己统一，但往往统一的是主干，不是输入界面；统一的是理解，不是生成；统一的是高层语义，不是低层 token 世界。Chameleon 则把统一推进到了一个非常前的位置。

它的美感在于架构整齐：

- 一个 token interface，
- 一个 AR backbone，
- 一个 next-token objective，
- 一个 mixed-modal document view。

它的代价也同样整齐：一旦视觉 tokenizer 不够强，或者 AR 采样过慢，这条路线的短板会立刻暴露出来。也正因为它足够纯，后续很多混合路线都可以看成是在回答 Chameleon 留下的问题。

## 与相邻方法的关系

### 对比 Emu3

Emu3 与 Chameleon 最像，二者都高度信任 token 统一与 AR 建模。Emu3 更进一步把视频明确纳入统一叙事，因此可以看成是 Chameleon 路线在模态范围上的进一步外推。

### 对比 Janus

Janus 认为视觉入口不应该过度统一，而需要在理解与生成之间做编码解耦。Chameleon 恰好站在另一端：它主张尽可能早地统一图像和文本，因此二者可以看作 unified model 设计上的两极。

### 对比 Show-o

Show-o 共享 transformer 主干，但图像生成改用离散去噪式 mask prediction。Chameleon 则连图像生成动力学也尽量统一到 AR next-token prediction 中，因此更纯粹，但也更依赖视觉 token 化的质量与效率。

### 对比 Orthus / Transfusion

Orthus、Transfusion 都代表“共享主干，保留模态专属输出动力学”的路线。Chameleon 则更像在说：如果你真相信 foundation model 范式，就应该尽量少留专属路径。

## 重要细节

- Architecture: mixed-modal early-fusion token-based AR transformer
- Objective: 统一混合序列上的 autoregressive next-token prediction
- Representation: 文本 token 与离散视觉 token 的单一 token 流
- Data view: 文本、图像、图文交错文档统一为 mixed-modal sequence
- Strengths: 统一程度高；接口非常纯；图文任意交错生成很自然；foundation model 叙事完整
- Limitations: 强依赖 image tokenizer；图像生成质量和采样效率受 token 路线限制；稳定训练难度高

## 我的笔记 / 开放问题

### 我的笔记

我觉得 Chameleon 最重要的意义，不只是它“能做 mixed-modal generation”，而是它把 unified model 的问题提得足够干净。很多后来方法之所以要保留模态专属路径、diffusion head、flow head，恰恰是因为 Chameleon 这种纯 AR early-fusion 路线把问题暴露得太清楚了：统一当然优雅，但优雅的代价通常是视觉离散化和序列建模被逼到极限。

换句话说，Chameleon 像是 unified token route 的一个非常标准的原点。你可以不同意它的路线，但后续很多路线都必须先回答：为什么不直接像 Chameleon 那样做？

### 开放问题

- image tokenizer 的能力上限，会不会决定这类 early-fusion unified model 的最终上限？
- 纯 AR 图像生成在高分辨率下的采样成本，是否注定让这条路线在实用系统里受限？
- 当 mixed-modal 文档更长、更复杂时，统一 token 流是否还能维持稳定语义和视觉一致性？
- 如果未来视觉离散化进一步提升，Chameleon 这条看似“过于理想化”的路线会不会重新变得更有竞争力？

## 相关笔记

- [Emu3 笔记](./emu3-notes.md)
- [Janus 笔记](./janus-notes.md)
- [Orthus 笔记](./orthus-notes.md)
- [Show-o 笔记](./show-o-notes.md)

## 参考资料

- Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models", arXiv, 2024. https://arxiv.org/abs/2405.09818
- Hugging Face paper page. https://huggingface.co/papers/2405.09818

