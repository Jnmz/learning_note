# Transfusion 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

## 一句话总结

Transfusion 的核心洞见是：统一模型不必把所有模态都压成同一种变量形式，文本仍然适合 next-token prediction，图像仍然更适合 diffusion；真正值得统一的，是跨模态上下文建模的 transformer 主干，以及让离散文本 token 与连续视觉 token 在同一个语义空间里交互的条件机制。

## 背景 / 问题设定

Transfusion 面向的是 unified model 里的两种极端路线：

1. 全部离散化、全部自回归。这样做最统一，但图像生成往往失去 diffusion 类方法在视觉保真上的强归纳偏置。
2. 语言模型和图像扩散模型彼此独立，只在接口层简单拼接。这样图像生成强，但统一上下文建模能力又很弱。

论文的判断是：这两种极端都抓住了一半真相，却都不够完整。

- 纯 AR 路线抓住了“共享上下文接口”的优势；
- diffusion 路线抓住了“图像生成动力学与语言不同”的现实。

于是更核心的问题变成：

- 能否共享一个 transformer 主干？
- 同时又让文本和图像保留各自最自然的输出动力学？
- 如果文本是离散 token、图像是连续 patch 或 latent，模型是否还能原生统一？

Transfusion 给出的回答非常有代表性：统一不一定意味着所有模态服从同一个变量空间，而可以意味着

\[
\text{shared contextual reasoning backbone}
\quad + \quad
\text{modality-specific generation dynamics}.
\]

这也是它在 unified model 谱系中的独特位置。它不像 Emu3/Chameleon 那样试图统一到 token 世界，也不像传统系统那样完全拆开理解和生成，而是在中间找到了一个很强的结构平衡点。

## 记号

设：

- 文本 token 序列为 \(x = (x_1,\dots,x_m)\)
- 图像连续 patch / latent 序列为 \(y = (y_1,\dots,y_n)\)
- 扩散中的噪声时刻为 \(t \in [0,1]\)
- 加噪后的图像表示为 \(y_t\)
- 共享 transformer 主干为 \(f_\theta\)
- 文本语言模型头为 \(h_{\text{LM}}\)
- 图像 diffusion 头为 \(h_{\text{Diff}}\)
- 文本目标为 \(\mathcal{L}_{\text{LM}}\)
- 图像 diffusion 目标为 \(\mathcal{L}_{\text{Diff}}\)

Transfusion 的高层结构可以抽象写成

\[
h = f_\theta(z),
\]

其中 \(z\) 表示混合后的多模态上下文序列。根据当前目标 token 的模态类型，输出再分别走

\[
h_{\text{LM}}(h)
\quad \text{or} \quad
h_{\text{Diff}}(h, t).
\]

## 核心思想

### 1. 同一主干处理离散与连续模态

Transfusion 的统一点不在“单一 token space”，而在“单一 transformer 主干”。文本和图像都通过共享主干组织条件关系：

- 文本以离散 token 进入；
- 图像以连续 patch / latent 进入；
- 主干在同一上下文窗口中同时处理它们。

因此它统一的是上下文语义空间，而不是输入变量形式。

### 2. 文本走 AR，图像走 diffusion

这是方法最核心的选择。Transfusion 明确承认不同模态的最佳生成偏置不同：

- 文本天然适合 next-token prediction；
- 图像更适合 diffusion 或 flow 类连续去噪动力学。

因此它不追求“一个目标统治一切”，而是追求“一个主干承接一切条件关系”。

### 3. 统一主干，模态专属 I/O 并不削弱统一

论文里一个很关键的观念是：模态专属输入层和输出层不等于“不统一”。它们负责的是模态接口适配，而不是跨模态认知本体。

也就是说，Transfusion 重新定义了 unified model：

\[
\text{unified} \neq \text{same input/output variable type},
\]

而更接近于

\[
\text{unified} = \text{shared contextual computation}.
\]

## 一个简单示意图

```text
text tokens ------------------> text embedding ----------------+
                                                              |
image patches / latents ------> continuous image adapter -----+--> shared transformer
                                                              |
                                                              +--> LM head --------> next text token
                                                              |
                                                              +--> diffusion head -> image denoising / velocity / noise prediction
```

## 详细推导

### 推导 1：文本侧仍然是标准条件语言模型

若目标是根据多模态上下文生成文本，设共享主干看到的条件上下文为 \(c\)，那么文本分布仍然按标准自回归方式分解：

\[
p_\theta(x \mid c)
=
\prod_{i=1}^{m} p_\theta(x_i \mid x_{<i}, c).
\]

对应负对数似然损失为

\[
\mathcal{L}_{\text{LM}}
=
- \sum_{i=1}^{m} \log p_\theta(x_i \mid x_{<i}, c).
\]

这里真正重要的点不在公式本身，而在于 Transfusion 没有因为“统一”就去破坏文本最成熟的建模形式。它保留了语言模型最核心的因果 next-token 接口，因此：

\[
\text{text side} = \text{standard LM as much as possible}.
\]

这样一来，统一模型的难点就被集中到“如何让主干同时服务图像 diffusion”，而不是重写文本建模。

### 推导 2：图像侧通过扩散路径建模连续变量

与纯 AR 路线不同，Transfusion 不把图像硬离散成 token 后逐个预测，而是保留连续视觉表示 \(y\)，并在扩散路径上学习去噪或速度预测。

用标准扩散 / flow-matching 风格记号，可以先写出一条噪声路径。若 \(y_0\) 为噪声、\(y_1\) 为真实图像表示，则可抽象地写成

\[
y_t = \alpha_t y_1 + \sigma_t y_0.
\]

或者在线性 flow matching 视角下写成

\[
y_t = (1-t)y_0 + t y_1.
\]

Transfusion 的关键不在于必须选哪一种具体噪声参数化，而在于：图像头预测的是连续空间中的去噪信号，而不是离散类别。

若以最常见的噪声预测形式表示，目标可写成

\[
\mathcal{L}_{\text{Diff}}
=
\mathbb{E}_{y_1, y_0, t}
\left[
\left\|
\epsilon_\theta(y_t, t, c) - y_0
\right\|_2^2
\right],
\]

其中 \(c\) 由共享主干提供条件上下文。

这说明 Transfusion 真正共享的不是输出变量，而是图像 diffusion 所依赖的条件表示。

### 推导 3：联合目标体现的是“共享条件建模，不共享输出分布”

把两边放在一起，Transfusion 的总目标就是

\[
\mathcal{L}
=
\lambda_{\text{LM}} \mathcal{L}_{\text{LM}}
+
\lambda_{\text{Diff}} \mathcal{L}_{\text{Diff}}.
\]

表面上看这只是 loss 加权求和，但它实际上表达了一个更深的分解：

\[
\text{shared part} = f_\theta(\text{context}),
\qquad
\text{text output} \neq \text{image output}.
\]

也就是说，Transfusion 不试图拟合一个完全统一的输出分布，而是拟合：

- 文本侧的离散条件分布，
- 图像侧的连续去噪条件分布，

并让这两类分布共享同一个条件语义核心。

这和 Chameleon / Emu3 的关键差异就在这里。后者更接近

\[
\text{one variable type} \Rightarrow \text{one objective family},
\]

而 Transfusion 更接近

\[
\text{one contextual core} \Rightarrow \text{multiple objective families}.
\]

### 推导 4：为什么共享主干可能成立

Transfusion 的关键假设是：虽然文本和图像输出动力学不同，但它们共享大量条件依赖结构。设上下文为 \(c\)，则文本和图像分别依赖

\[
p(x \mid c), \qquad p(y \mid c).
\]

如果 \(c\) 只是在“模态适配层”之后才分别形成，那么跨模态共享会很弱；但若 \(c\) 来自一个共享主干

\[
c = f_\theta(\text{text context}, \text{image context}),
\]

那么语言理解、图文对齐、语义条件组织等能力就可以在这个共享空间里共同学习。

换句话说，Transfusion 把问题拆成两部分：

\[
\text{how to understand the mixed-modal context}
\]

和

\[
\text{how to realize the output in each modality}.
\]

它认为第一部分值得共享，第二部分不必强行共享。这就是它之所以成立的核心逻辑。

## 架构理解

### 1. 为什么它不是简单“两模型拼接”

如果只是“一个 LLM + 一个 diffusion model”，再在外层做接口编排，那不算 Transfusion 的核心思想。Transfusion 强调的是：

- 文本和图像都进入同一个主干参与条件建模；
- 共享主干直接承担跨模态上下文组织；
- diffusion 头不是外部黑盒，而是共享语义主干的输出分支。

因此它更像一个真正统一的多模态网络，而不是系统编排。

### 2. 为什么说它统一的是上下文，而不是变量

很多 unified model 讨论容易把“统一”理解成：

- 同一种 tokenizer，
- 同一种 token，
- 同一种 objective。

Transfusion 的重要贡献，是把统一的焦点移到更本质的地方：

- 真正需要共享的是跨模态上下文推理；
- 不同模态最终输出是否同构，未必是最重要的问题。

这让它成为 unified model 讨论中一个非常关键的“重新定义者”。

### 3. 推理时其实是在切换两种动力学

Transfusion 的推理并不是单一序列生成循环，而是会根据当前目标模态切换行为：

- 输出文本时，按语言模型方式自回归采样；
- 输出图像时，按 diffusion 过程迭代去噪。

因此它的统一更多是“共享脑干，切换手脚”，而不是“所有动作都由同一条神经回路完成”。

## 训练流程

根据论文和公开材料，Transfusion 的训练重点不在于复杂网络分支，而在于如何让共享主干同时吸收语言建模信号与图像 diffusion 信号。

### 阶段 1：保住纯文本能力

和很多 unified model 一样，Transfusion 需要大量纯文本训练来维持语言模型能力，否则 diffusion 支路很可能把主干往视觉条件网络方向拉偏。

### 阶段 2：加入图文与图像生成数据

随后引入图文和图像生成相关数据，使共享主干逐步学会：

- 文本条件如何组织图像语义；
- 图像条件如何辅助文本输出；
- 连续视觉去噪所需的条件上下文如何在主干中形成。

### 阶段 3：在规模上验证混合路线是否稳定

Transfusion 特别强调 scaling behavior，这说明它真正关心的不只是“小规模能跑”，而是：

- 参数规模增大后混合目标是否还能稳定；
- 主干是否会被两类梯度撕扯；
- hybrid unified route 是否在大规模下比极端路线更合理。

这也是它区别于很多“结构上很漂亮但规模证据不足”的方法的地方。

## 直觉 / 理解

我对 Transfusion 的理解是：它像 unified model 领域里的一个“结构现实主义者”。纯 AR 路线很整齐，但图像生成往往不够自然；纯 diffusion 或外挂生成路线图像很好，但统一性不够。Transfusion 的态度是，没必要把这两种优点二选一。

它最有价值的地方，是把 unified model 的核心从“所有模态都做同一件事”改成了“所有模态在同一个认知核心里交互，但保留各自最自然的实现动力学”。这让 unified model 的讨论一下子从“形式整齐”推进到了“功能上真正该共享什么”。

## 与相邻方法的关系

### 对比 Emu3 / Chameleon

Emu3、Chameleon 更相信只要 token 接口统一，自回归目标就可以覆盖一切。Transfusion 正好站在反面：它认为图像生成不必为了统一而放弃 diffusion family 的归纳偏置，因此统一应止步于主干而不是输出动力学。

### 对比 Orthus

Orthus 也体现“共享主干 + 模态专属输出”的思想，但它仍保留很强的 AR 路线背景，更像是在 AR 统一范式内部做修补。Transfusion 的 diffusion 立场更明确，也更系统化。

### 对比 Show-o / Show-o2

Show-o 系列是在统一主干里组合文本 AR 和视觉离散去噪 / flow matching。Transfusion 与它们相似之处在于都承认模态专属动力学的重要性；不同在于 Transfusion 更直接地从“文本离散、图像连续”的变量差异出发组织整个方法。

### 对比 TUNA

TUNA 进一步把焦点前移到 unified visual representation，本质上是在问：若视觉表征空间本身不统一，主干共享是否足够？Transfusion 则更关注：在已有视觉连续表示前提下，如何让共享主干与 diffusion 最自然地协作。

## 重要细节

- Architecture: 共享 transformer 主干 + 模态专属输入层 / 输出层
- Objective: 文本 next-token prediction + 图像 diffusion / continuous denoising objective
- Representation: 文本离散 token 与图像连续 patch / latent 共存于同一上下文主干
- Data: 纯文本、图文混合、图像生成相关训练数据
- Evaluation: text-only、cross-modal benchmarks、image generation、scaling behavior
- Strengths: 不强迫统一变量形式；统一逻辑清晰；对 hybrid route 的规模行为有明确关注
- Limitations: 训练和推理比纯 AR 更复杂；视觉分支仍保留 diffusion 采样成本；共享主干可能面临更复杂的梯度权衡

## 我的笔记 / 开放问题

### 我的笔记

我觉得 Transfusion 最重要的贡献，是把 unified model 的讨论从“同构崇拜”里拉出来。它提醒我们，统一不一定意味着所有模态都要变成一种 token、一种 loss、一种生成循环；真正值得统一的，也许是推理和条件组织的核心，而不是输出变量形式本身。

这使它在方法史上特别重要，因为很多后续 hybrid unified model 都在不同程度上继承了这种思路：共享主干，但不强迫图像像语言一样工作。

### 开放问题

- 在更大规模下，共享主干到底会更偏向语言建模，还是会被视觉 diffusion 信号显著改变？
- 文本和图像梯度在同一主干中的冲突，是否需要更细粒度的 routing 或 loss balancing？
- 若视觉表征空间进一步改进，Transfusion 这种 hybrid route 的收益会继续扩大，还是会被更原生 unified representation 方法替代？
- 当视频也被纳入时，这种“共享主干 + 模态专属动力学”的框架是否还能保持同样的简洁性？

## 相关笔记

- [Emu3 笔记](./emu3-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)
- [Orthus 笔记](./orthus-notes.md)
- [TUNA 笔记](./tuna-notes.md)

## 参考资料

- Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", arXiv, 2024. https://arxiv.org/abs/2408.11039
- Hugging Face paper page. https://huggingface.co/papers/2408.11039

