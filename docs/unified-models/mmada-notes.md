# MMaDA 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - MMaDA: Multimodal Large Diffusion Language Models
  - MMaDA official repository

## 一句话总结

MMaDA 的重要性在于，它尝试把统一模型的中心动力学从 autoregressive 迁移到 diffusion，并进一步把这种统一从预训练延伸到 mixed long CoT 微调与 UniGRPO 强化学习阶段；也就是说，它不是“给 diffusion 接一个多模态壳”，而是试图把 diffusion 提升成完整的 multimodal foundation model 范式。

## 背景 / 问题设定

到 2025 年前后，统一模型虽然已经很活跃，但主叙事仍然普遍围绕 LLM 组织：

- 文本侧默认 AR；
- 视觉侧通常是附属分支、辅助头或外挂生成器；
- 即使在 unified model 里，真正的“认知中心”往往还是语言模型。

MMaDA 提出一个更激进的问题：

- diffusion 能否不只是图像生成器，而是更一般的统一 foundation model 形式？
- 如果统一真的围绕 diffusion 展开，文本理解、多模态理解、图像生成是否还能共享同一个主干？
- 若中心动力学从 next-token prediction 改成统一扩散过程，那么微调、长链推理和强化学习又该怎么配套？

因此它关心的不是“把 diffusion 接到 LLM 上”，而是：

\[
\text{can diffusion itself become the core language of a unified model?}
\]

这和 Transfusion、Show-o2 这类混合路线不同。后者通常仍保留一个强语言模型核心，再为视觉接上 diffusion / flow 头；MMaDA 则更进一步，试图把统一的中心从 AR-LM 彻底迁移到 diffusion backbone。

## 记号

设：

- 多模态状态序列或统一状态表示为 \(x\)
- 扩散时间步为 \(t \in [0,1]\) 或离散时间索引 \(t \in \{1,\dots,T\}\)
- 干净样本为 \(x_1\)
- 噪声样本为 \(x_0\)
- 加噪后的中间状态为 \(x_t\)
- 统一 diffusion backbone 为 \(f_\theta\)
- 统一 diffusion 目标为 \(\mathcal{L}_{\text{diff}}\)
- 长链推理微调目标为 \(\mathcal{L}_{\text{CoT}}\)
- 强化学习阶段目标为 \(\mathcal{L}_{\text{RL}}\)

MMaDA 的核心高层关系可以抽象为

\[
x_t = \alpha_t x_1 + \sigma_t x_0,
\]

以及

\[
f_\theta(x_t, t, c),
\]

其中 \(c\) 表示任务条件、文本上下文、视觉上下文或混合多模态提示。

## 核心思想

### 1. 用 diffusion 统一 foundation model

MMaDA 不把 diffusion 仅仅看成图像生成子模块，而是把它提升为统一模型的基本概率建模形式。

这意味着在它的视角里，真正应当统一的不是“下一个 token”，而是跨模态状态从噪声到数据的演化过程。

### 2. 模态无关而不是模态拼接

论文强调 modality-agnostic design，意思不是模态消失了，而是模型试图让文本、多模态理解和图像生成尽量共享同一种训练叙事，而不是：

- 先有 LLM，
- 再补一个视觉模块，
- 再补一个图像生成器，
- 最后再在系统层面拼起来。

MMaDA 试图避免这种“功能叠加式统一”，转而用统一 diffusion backbone 贯穿不同模态。

### 3. 统一预训练与后训练

MMaDA 很重要的一点，是它不只提出一个预训练骨架，还把 mixed long CoT fine-tuning 和 UniGRPO 强化学习纳入同一研究路径。

也就是说，它的 ambition 不是“证明 diffusion backbone 也能跑预训练”，而是：

\[
\text{pretraining}
\rightarrow
\text{reasoning fine-tuning}
\rightarrow
\text{RL alignment}
\]

都围绕 diffusion foundation model 重新组织。

## 一个简单示意图

```text
text / multimodal prompt
        |
        v
unified noisy multimodal state
        |
        v
shared diffusion backbone
        |
        +--> understanding / reasoning denoising trajectory
        |
        +--> image generation trajectory
```

## 详细推导

### 推导 1：MMaDA 把统一目标从 next-token prediction 改成统一扩散过程

AR 统一模型通常写成

\[
p_\theta(s) = \prod_{i=1}^{N} p_\theta(s_i \mid s_{<i}),
\]

而 MMaDA 想做的不是拟合这个因果分解，而是拟合统一状态从噪声走向数据的扩散 / 去噪过程。

设真实多模态状态为 \(x_1\)，噪声样本为 \(x_0\)，则一个标准连续路径可写成

\[
x_t = \alpha_t x_1 + \sigma_t x_0.
\]

若采用更接近 flow matching 的线性记号，也可写成

\[
x_t = (1-t)x_0 + t x_1.
\]

模型不再预测“下一个 token 是什么”，而是预测：

- 噪声 \(\epsilon\)，
- score，
- 或 velocity / denoised state，

本质上是在学习一个统一状态动力学。

因此 MMaDA 的一个核心思想可以抽象成：

\[
\text{unified modeling}
\neq
\text{unified token sequence},
\]

而更接近于

\[
\text{unified stochastic dynamics over multimodal states}.
\]

### 推导 2：统一 diffusion 损失体现 modality-agnostic 建模

若以最常见的噪声预测方式表述，MMaDA 的基础目标可以写成

\[
\mathcal{L}_{\text{diff}}
=
\mathbb{E}_{x_1, x_0, t}
\left[
\left\|
\epsilon_\theta(x_t, t, c) - x_0
\right\|_2^2
\right].
\]

这里关键不在于这条公式本身新不新，而在于 \(x_1\) 不再只代表图像，而可以代表：

- 文本状态，
- 多模态理解状态，
- 图像生成状态，
- 混合模态推理状态。

因此所谓 modality-agnostic，不是说不同模态完全没有差异，而是说它们尽量被投到同一种扩散训练叙事里：

\[
\text{text}, \text{vision}, \text{reasoning}
\rightarrow
\text{one diffusion objective family}.
\]

这和 Transfusion / Orthus 的关键区别在于：后者仍保留“文本 AR，图像 diffusion”的双动力学；MMaDA 更想要一套统一动力学覆盖不同能力。

### 推导 3：Mixed Long CoT 微调意味着 diffusion backbone 也要承载推理轨迹

MMaDA 很重要的一点，是它不满足于“能做生成”，而要让 diffusion foundation model 也能做长链 reasoning。于是第二阶段目标可以抽象写成

\[
\mathcal{L}_{\text{stage2}}
=
\mathcal{L}_{\text{diff}}
+
\lambda \mathcal{L}_{\text{CoT}}.
\]

这里的关键不是把 CoT 当作普通监督信号相加，而是把 reasoning 也放进 diffusion 范式中重新理解：

\[
\text{reasoning}
\approx
\text{structured denoising / refinement trajectory}.
\]

换句话说，MMaDA 不是简单把“推理”外挂到 diffusion 模型后面，而是尝试让 diffusion backbone 本身学会生成更长、更稳定、更可控的推理轨迹。

这也是它和很多 diffusion-only 视觉模型非常不同的地方：它不是只追求画图，而是要把 reasoning 纳入同一个 backbone 的能力范围。

### 推导 4：UniGRPO 说明作者想补齐 diffusion unified model 的后训练闭环

第三阶段的关键在于 RL。若统一 diffusion 模型也要像 LLM 一样具备可对齐、可偏好优化的能力，那么后训练目标就不能停在监督微调。

抽象地写，第三阶段可表示为

\[
\mathcal{L}_{\text{stage3}} = \mathcal{L}_{\text{RL}}.
\]

更具体地说，UniGRPO 的意义不是某个单独公式本身，而是它传递了一个更强的系统判断：

\[
\text{a diffusion unified model also needs its own RL recipe}.
\]

这非常重要，因为它意味着 MMaDA 不是只在讨论一个新 backbone，而是在尝试补齐 diffusion unified model 的完整训练栈：

- 预训练怎么做；
- reasoning 微调怎么做；
- alignment / RL 怎么做。

这使它更像一个范式提案，而不只是一个架构 patch。

## 架构理解

### 1. 为什么它不是“图像 diffusion + 文本补丁”

如果只是一个图像 diffusion 模型，额外接一点文本功能，那不算 MMaDA 的核心。MMaDA 真正强调的是：

- diffusion backbone 本身要承载文本与多模态能力；
- 统一主干不是 AR-LM；
- 文本和图像都要纳入同一扩散训练叙事。

因此它不是“给 diffusion 模型加语言接口”，而是“把 diffusion 提升为 language-model-scale 的统一 backbone”。

### 2. 为什么它在 unified model 谱系里很激进

MMaDA 的激进之处在于，它挑战的不是某个局部设计，而是整个范式中心：

- 为什么 unified model 的中心必须是 AR-LM？
- 为什么 reasoning 和 RL 只能围绕 token-based policy 来组织？

它其实是在问：如果把 foundation model 的核心对象从 token chain 换成 diffusion dynamics，会发生什么。

### 3. 为什么它特别重视完整训练栈

很多方法会提出一个新 backbone，但并没有说明这个 backbone 如何承担后训练、长链推理、偏好优化。MMaDA 特别强调 mixed long CoT 与 UniGRPO，说明作者知道：

\[
\text{a foundation model is not complete without a post-training story}.
\]

这也是它相对不少“只停留在预训练”的 diffusion unified work 更完整的地方。

## 训练流程

MMaDA 的训练适合按阶段理解，而不是只盯某个单一 loss。

### 阶段 1：Unified Diffusion Pretraining

第一阶段在统一 diffusion 目标下学习文本、多模态理解和图像生成基础能力：

\[
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{diff}}.
\]

这一阶段的重点，是先证明一个共享 diffusion backbone 可以承载多模态基础建模。

### 阶段 2：Mixed Long CoT Fine-Tuning

第二阶段引入更偏 reasoning 的长链数据，让 diffusion 基座不仅能生成，也能形成更稳定的推理表达：

\[
\mathcal{L}_{\text{stage2}}
=
\mathcal{L}_{\text{diff}}
+
\lambda \mathcal{L}_{\text{CoT}}.
\]

这一阶段的重点，是把“能生成”推进到“能推理”。

### 阶段 3：UniGRPO Reinforcement Learning

第三阶段用统一 RL 算法对 diffusion foundation model 做后训练：

\[
\mathcal{L}_{\text{stage3}} = \mathcal{L}_{\text{RL}}.
\]

这一步的目标不是再提一点指标，而是补齐 diffusion unified model 的 alignment 能力。

## 直觉 / 理解

我对 MMaDA 的理解是：它像 unified model 领域里一次很明确的“范式挑战”。很多 unified model 讨论，默认前提都是“语言模型当然该是中心，视觉只是如何接进去的问题”。MMaDA 则反过来问：为什么中心不能是 diffusion？

它最有价值的地方，不是某一项单点性能，而是逼着我们重新思考 unified model 的基础对象到底应该是什么：

- 是 token chain，
- 是共享主干加模态专属动力学，
- 还是更一般的概率状态演化？

MMaDA 给出的答案明显偏向第三种。

## 与相邻方法的关系

### 对比 Emu3 / Chameleon

Emu3、Chameleon 认为统一应围绕 token-based AR 展开。MMaDA 几乎站在对立面：它认为统一也可以围绕 diffusion dynamics 建立，因此二者代表 unified model 里最鲜明的两种哲学。

### 对比 Transfusion / Orthus

Transfusion、Orthus 都属于“共享主干 + 模态专属动力学”的混合路线。MMaDA 更进一步，它想把统一动力学本身也尽量迁移到 diffusion，而不是停在 hybrid compromise。

### 对比 Show-o2

Show-o2 已经把视觉侧推向 latent flow matching，并尝试扩展到 text-image-video。MMaDA 与它相似之处在于都不再认为 AR 是唯一中心；不同在于 MMaDA 更强调“扩散范式本身也应覆盖后训练和 RL 栈”。

### 对比 LLaDA-o

LLaDA-o 与 MMaDA 最像，二者都属于 diffusion-centered / omni-diffusion unified modeling 家族。MMaDA 相对更强调完整训练栈和后训练闭环，而不只是统一生成骨架。

## 重要细节

- Architecture: 统一 diffusion foundation model
- Objective: diffusion 预训练 + mixed long CoT fine-tuning + UniGRPO 后训练
- Design: modality-agnostic unified diffusion backbone
- Data: 文本推理、多模态理解、图像生成与 CoT 数据
- Evaluation: textual reasoning、multimodal understanding、text-to-image generation
- Strengths: 重新定义统一中心动力学；很重视后训练完整性；不是只讲预训练架构
- Limitations: diffusion unified model 生态仍新；文本侧效率和交互性仍待进一步验证；与 AR-LM 生态相比工具链还不成熟

## 我的笔记 / 开放问题

### 我的笔记

我觉得 MMaDA 的意义很像一次“把问题重新提一遍”。很多方法都在讨论怎么把视觉接入语言模型，但 MMaDA 更像在问：统一模型为什么非得以语言模型为中心？如果这个问题成立，那它对 unified model 方向的影响会很深，因为它挑战的是整个研究默认前提。

这也意味着，MMaDA 的价值未必只体现在今天的指标上，而更体现在它是否能催生一整套 diffusion foundation model 的工程生态。

### 开放问题

- diffusion 作为统一 backbone，在长文本、高频交互和工具使用等方向上，是否真能建立与 AR-LM 同等成熟的生态？
- mixed long CoT 在 diffusion 范式下究竟学到的是“推理”，还是更强的结构化去噪？
- UniGRPO 这类 RL 方法在 diffusion foundation model 上会不会遇到与 token policy 完全不同的稳定性问题？
- 若未来 omni-diffusion 方法继续发展，MMaDA 这种路线会不会真正把 unified model 的中心从 LLM 挪走？

## 相关笔记

- [LLaDA-o 笔记](./llada-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)
- [Emu3 笔记](./emu3-notes.md)

## 参考资料

- Yang et al., "MMaDA: Multimodal Large Diffusion Language Models", arXiv, 2025. https://arxiv.org/abs/2505.15809
- Official repository. https://github.com/Gen-Verse/MMaDA
- Hugging Face paper page. https://huggingface.co/papers/2505.15809

