# Show-o2 Notes

## Metadata

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Show-o2: Improved Native Unified Multimodal Models
  - Show-o official repository and model cards

## One-Sentence Takeaway

Show-o2 可以看成是对 Show-o 的一次“原生升级”：它不再把视觉统一建立在离散图像 token 上，而是把图像和视频统一到 3D causal VAE latent space，再用 language head 做文本自回归、用 flow head 做视觉 flow matching。

## Background / Problem Setup

Show-o 已经证明了一个 transformer 主干可以同时覆盖理解和生成，但它还留着几个明显问题：

- 图像生成建立在离散 token 上，视觉细节和高保真生成上限受 tokenizer 影响较大。
- 方法主要围绕 text + image，向 video 扩展并不自然。
- 视觉理解和视觉生成需要的表征并不完全相同，单一路径视觉表示容易顾此失彼。

Show-o2 的核心判断是：

1. 统一模型不一定要统一到“离散图像 token”这一层。
2. 更自然的统一空间可以是连续的 3D causal VAE latent。
3. 文本与视觉可以继续共用一个语言模型主干，但需要专门的输出头和更细致的视觉表示构造。

因此 Show-o2 的升级不是“小修小补”，而是把视觉侧从 discrete diffusion 迁移到了 latent flow matching，并把统一范围扩展到了 text-image-video。

## Notation

设：

- 文本 embedding 序列为 \(e = (e_1, \dots, e_T)\)
- 图像或视频经 3D causal VAE 编码后的 latent 为 \(z\)
- 噪声时刻 \(t \in [0,1]\)
- 语义路径提取的高层特征为 \(h_{\text{sem}}(z_t)\)
- 投影路径提取的低层特征为 \(h_{\text{low}}(z_t)\)
- 融合后的统一视觉表示为 \(u\)
- 语言头输出的文本条件分布为 \(p_\theta(\cdot)\)
- flow head 预测的速度场为 \(v_\theta(\cdot, t)\)

## Core Idea

### 1. Move From Discrete Image Tokens To 3D Causal VAE Latents

Show-o2 直接在 3D causal VAE 空间里处理视觉信号。这样做的好处有两个：

- 图像和视频都能被同一种 latent 表示容纳
- 视觉生成不再受离散 codebook 的硬量化误差约束

这里的“3D”不是指 3D 场景，而是指 latent 结构显式保留时间维，因此同一套编码器既能表示单帧图像，也能表示视频片段。

### 2. Dual-Path Visual Representation

论文认为理解与生成对视觉特征的偏好不同：

- 理解更需要高层语义
- 生成更需要低层纹理、结构、文字细节

所以 Show-o2 不再只走一条视觉路径，而是同时保留：

- semantic layers: 偏语义、上下文化
- projector path: 偏低层、尽量保真

然后通过 spatial / temporal fusion 把两路特征合成统一视觉表示。

### 3. Two Heads, Two Dynamics

统一仍然发生在同一个 language model 主干里，但输出机制被明确拆开：

- language head 负责文本 token prediction
- flow head 负责图像 / 视频 latent 的速度预测

这意味着 Show-o2 的“统一”比 Show-o 更彻底也更模块化：主干统一，输出动力学不统一。

## A Simple Diagram

```text
text ----------> text embeddings -----------------------------+
                                                             |
image/video -> 3D causal VAE latents -> dual-path fusion ----+--> shared LM backbone
                                                                     |
                                                                     +--> language head -> autoregressive text
                                                                     |
                                                                     +--> flow head -> latent flow matching
```

## Detailed Derivation

### Derivation 1: Unified Visual Representation Is A Fusion Of Two Complementary Views

Show-o2 的第一步不是直接把视觉 latent 扔给 LLM，而是先构造 unified visual representation。设输入图像或视频经 3D causal VAE 后得到 latent \(z\)，在噪声时刻 \(t\) 上得到 \(z_t\)。

论文中的双路径思想可以写成：

\[
h_{\text{sem}} = h_{\text{sem}}(z_t),
\qquad
h_{\text{low}} = h_{\text{low}}(z_t).
\]

其中：

- \(h_{\text{sem}}\) 由 semantic layers 提取，更强调语义上下文
- \(h_{\text{low}}\) 由 projector 路径提取，更强调底层结构和细节

接着通过 spatial (-temporal) fusion 得到统一表示

\[
u = \operatorname{STF}\big([h_{\text{sem}} ; h_{\text{low}}]\big),
\]

这里 \([\,\cdot\, ; \,\cdot\,]\) 表示在特征维拼接，\(\operatorname{STF}\) 表示论文里的 spatial / temporal fusion 模块。

为什么这种两路融合是必要的？

如果只保留语义路径，那么模型对问答和描述会更有利，但生成时容易丢纹理、排版和局部结构。反过来，如果只保留低层路径，生成保真度可能更好，但理解时可供推理的抽象语义不足。

所以 Show-o2 这里隐含的是一种表示分解假设：

\[
\text{good representation}
\neq
\text{pure semantics}
\neq
\text{pure low-level details},
\]

而更接近于

\[
\text{good representation}
\approx
\text{semantic abstraction} + \text{structural fidelity}.
\]

这也是它相对 Show-o 最重要的方法升级之一，因为 Show-o 默认更依赖单一视觉离散表示。

### Derivation 2: Text Objective Still Follows Autoregressive Factorization

虽然 Show-o2 大改了视觉侧，但文本侧依然保持 LLM 的标准形式。若输出文本 token 为 \(y = (y_1,\dots,y_T)\)，条件是文本前缀与统一视觉表示 \(u\)，则：

\[
p_\theta(y \mid u)
= \prod_{t=1}^{T} p_\theta(y_t \mid y_{<t}, u).
\]

于是负对数似然为

\[
\mathcal{L}_{\text{text}}
= - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, u).
\]

这个目标的意义在于，Show-o2 并没有因为“统一”而牺牲语言模型的基本接口。它仍旧把语言能力寄托在一个标准的 causal language modeling 目标上，因此可以最大化复用预训练 LLM 的参数与知识。

从架构角度看，这也解释了为什么它不直接把所有输出都改成 diffusion 或 flow：文本序列天然有顺序结构，而 AR factorization 已经是非常成熟的解。

### Derivation 3: Flow Matching Replaces Discrete Denoising On The Visual Side

论文将视觉生成交给 flow head，并明确说使用 flow matching 预测 velocity。论文正文没有把所有中间公式完全展开，因此下面用标准 flow matching 记号解释其含义。

设真实视觉 latent 为 \(x_1\)，噪声样本为 \(x_0 \sim \mathcal{N}(0, I)\)。标准 flow matching 会定义一条连接噪声与数据的插值路径，例如最常见的线性路径：

\[
x_t = (1-t)x_0 + t x_1, \qquad t \in [0,1].
\]

对 \(t\) 求导，得到对应的目标速度场：

\[
\frac{d x_t}{dt} = x_1 - x_0.
\]

如果模型预测速度 \(v_\theta(x_t, t, c)\)，其中 \(c\) 表示文本条件和上下文，那么标准的 flow matching 损失就是

\[
\mathcal{L}_{\text{flow}}
= \mathbb{E}_{x_0, x_1, t}
\left[
\| v_\theta(x_t, t, c) - (x_1 - x_0) \|_2^2
\right].
\]

这和 Show-o 的离散 mask prediction 有什么根本区别？

- Show-o 在离散 token 空间恢复被 mask 的类别
- Show-o2 在连续 latent 空间预测生成轨迹的速度

因此 Show-o2 的视觉生成建模从“分类式去噪”变成了“向量场回归”。这通常更适合高保真视觉生成，也更容易向视频推广，因为视频 latent 天然位于连续空间里。

### Derivation 4: The Native Unified Loss Is A Sum Of Language And Flow Terms

把上面两部分合起来，Show-o2 的原生统一训练可以概括成

\[
\mathcal{L}
= \mathcal{L}_{\text{text}} + \lambda \mathcal{L}_{\text{flow}}.
\]

展开后就是

\[
\mathcal{L}
=
- \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, u)
+ \lambda \,
\mathbb{E}_{x_0, x_1, t}
\left[
\| v_\theta(x_t, t, c) - (x_1 - x_0) \|_2^2
\right].
\]

这里的核心点不是“两个 loss 相加”本身，而是两种 loss 共用同一主干时所表达的归纳偏置：

- language head 要求主干保留因果语言建模能力
- flow head 要求主干提供足够好的跨模态条件表示，支撑视觉 trajectory learning

这实际上比 Show-o 更像真正的 native unified model，因为视觉生成不再借助离散 token reconstruction 的中介，而是直接在 latent dynamics 上建模。

## Training Recipe

### Stage 1: Pretrain The Visual Side Without Destroying Language Knowledge

论文特别强调，基础 LLM 自带语言知识，但不自带视觉生成能力。因此第一阶段只训练：

- projector
- spatial / temporal fusion
- flow head

而不是一开始就全模型一起更新。这样做的目的很明确：先把视觉生成能力“接”到主干上，同时尽量不破坏已有语言参数。

### Stage 2: Full-Model Unified Fine-Tuning

第二阶段再用高质量多模态理解数据与视觉生成数据联合微调全模型，让共享主干真正完成任务级统一。

### Why This Two-Stage Schedule Matters

如果一开始就让整个 LLM 为视觉生成大幅更新，而又没有同规模高质量文本语料去对冲，那么语言知识很容易退化。Show-o2 的两阶段训练，本质上是在做一个参数更新隔离：

- 第一阶段先学视觉适配器和生成头
- 第二阶段再适度释放全模型

这比 Show-o 的三阶段离散 token 训练更像“保住 LLM，外挂连续视觉生成能力，再回到统一微调”。

## Intuition / Interpretation

我对 Show-o2 的理解是：它把“统一模型”从接口统一推进到了动力学统一。

Show-o 的统一方式是：

- 主干统一
- 文本 AR
- 图像离散去噪

Show-o2 则进一步把视觉侧换成更接近现代生成模型主流范式的 latent flow matching，同时又不放弃同一个 LLM 主干。这让它更像一个真正的 any-to-any 原型系统。

另一个很有意思的点是 dual-path 表示。它其实承认了一个常被忽视的现实：用于“看懂”的视觉特征和用于“生成好”的视觉特征并不完全相同。Show-o2 不是回避这个冲突，而是显式用两条路径去吸收它。

## Relation to Other Methods

### Versus Show-o

Show-o 是离散图像 token + mask prediction；Show-o2 是 3D causal VAE latent + flow matching。后者更容易扩展到高保真图像与视频。

### Versus Pure MLLMs

纯 MLLM 通常只做理解，不具备强原生生成能力。Show-o2 的 ambition 更大，它要把理解、图像生成、视频生成、混合模态生成都放进一个主干里。

### Versus Transfusion / Emu3 / Chameleon

这些工作也都在探索 unified modeling，但 Show-o2 的独特点在于：

- 明确依赖 3D causal VAE 统一 image/video latent
- 用 dual-path 表示处理理解与生成的特征冲突
- 用 flow head 与两阶段训练，保护 LLM 的语言能力

## Important Details

- Architecture: 预训练 Qwen2.5 系列 LLM + language head + flow head + dual-path visual encoder side
- Objective: 文本 next-token prediction + 视觉 latent flow matching
- Data: 约 66M 图文对，并逐步加入 interleaved data、video-text pairs，再用 9M 高质量理解数据和 16M 高质量生成数据联合微调
- Evaluation: multimodal understanding、text-to-image、image/video generation、mixed-modality generation
- Strengths: 原生覆盖 text/image/video；视觉生成范式更现代；通过两阶段训练尽量保语言能力
- Limitations: 工程复杂度明显提高；对 3D causal VAE 和蒸馏语义层依赖更强；统一训练稳定性仍然是核心挑战

## My Notes / Open Questions

### My Notes

我觉得 Show-o2 比 Show-o 更像一个“研究方向宣言”。它在说，未来统一模型未必应该继续把所有视觉内容硬离散化，而可以把 LLM 当成跨模态 reasoning core，把连续视觉动力学当成并列的一等公民。

另一个值得注意的点是，它其实把视频也纳入了统一叙事。虽然今天很多 unified 模型还主要停留在图文层面，但 Show-o2 已经在架构层面为视频留好了位置。

### Open Questions

- dual-path fusion 的收益主要来自语义路径，还是主要来自低层保真路径？
- flow head 和共享主干之间是否会在大规模联合训练下产生更强的梯度冲突？
- 若不依赖 3D causal VAE，而改用更强的视频 tokenizer 或更强视觉 latent，效果会不会继续提升？

## See Also

- [Unified Multimodal Models Overview](./unified-multimodal-models-overview.md)
- [Show-o Notes](./show-o-notes.md)

## References

- Xie et al., "Show-o2: Improved Native Unified Multimodal Models", arXiv, 2025. https://arxiv.org/abs/2506.15564
- Official repository. https://github.com/showlab/Show-o
- Show-o2 model card. https://huggingface.co/showlab/show-o2-7B
