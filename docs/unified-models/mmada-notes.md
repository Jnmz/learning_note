# MMaDA 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - MMaDA: Multimodal Large Diffusion Language Models
  - MMaDA official repository

## 一句话总结

MMaDA 的重要性在于，它尝试把统一模型的中心动力学从 autoregressive 迁移到 diffusion，并进一步把这种统一从预训练延伸到 CoT 微调与强化学习阶段。

## 背景 / 问题设定

到 2025 年前后，统一模型仍然普遍围绕 LLM 组织：文本侧几乎默认 AR，视觉侧至多是附属分支。MMaDA 提出一个更激进的问题：

- diffusion 能否不只是图像生成器，而是更一般的统一 foundation model 形式？
- 如果统一真的围绕 diffusion 展开，后训练、推理和强化学习又该怎么配套？

因此它关心的不是“把 diffusion 接到 LLM 上”，而是“以 diffusion 为中心重写统一模型范式”。

## 记号

设：

- 多模态状态序列为 \(x\)
- 扩散时间步为 \(t\)
- 统一 diffusion 目标为 \(\mathcal{L}_{\text{diff}}\)
- 长链推理微调目标为 \(\mathcal{L}_{\text{CoT}}\)
- 强化学习阶段目标为 \(\mathcal{L}_{\text{RL}}\)

## 核心思想

### 1. 用 diffusion 统一 foundation model

MMaDA 不把 diffusion 仅仅看成图像生成子模块，而是把它提升为统一模型的基本概率建模形式。

### 2. 模态无关而不是模态拼接

论文强调 modality-agnostic design，意味着它试图让文本、多模态理解和图像生成尽量在同一训练叙事里出现，而不是“先有 LLM，再加多模态补丁”。

### 3. 统一预训练与后训练

MMaDA 不是只提出一个预训练骨架，还把 mixed long CoT fine-tuning 和 UniGRPO 强化学习纳入同一研究路径，试图让 diffusion foundation model 也拥有完整的后训练范式。

## 一个简单示意图

```text
text / multimodal input
        |
        v
unified diffusion representation
        |
        v
shared diffusion backbone
        |
        +--> understanding / reasoning output
        |
        +--> image generation trajectory
```

## Architecture / Data Flow

论文更偏系统描述而不是像传统 LLM 那样给出极细的 token-flow 图。按文中信息整理，MMaDA 的数据流更适合理解为“统一扩散过程中的多模态状态演化”。

它与 AR unified model 的差别在于：

- AR 模型学的是条件下一个 token 分布；
- MMaDA 学的是统一 diffusion dynamics。

所以它的共享不是共享一个语言状态机，而是共享一套跨模态扩散过程。文本、多模态理解与图像生成不是通过不同头简单拼接，而是在一个 diffusion backbone 中共享中间状态演化。

## Training Objective / Recipe

MMaDA 的训练适合按阶段理解：

### 阶段 1：Unified Diffusion Pretraining

第一阶段在统一 diffusion 目标下学习文本 / 多模态 / 图像生成基础能力：

\[
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{diff}}.
\]

### 阶段 2：Mixed Long CoT Fine-Tuning

第二阶段引入更偏 reasoning 的长链数据，目的是让 diffusion 基座不仅能生成，也能形成稳定推理表达：

\[
\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{CoT}}.
\]

### 阶段 3：UniGRPO Reinforcement Learning

第三阶段用统一 RL 算法对 diffusion foundation model 做后训练：

\[
\mathcal{L}_{\text{stage3}} = \mathcal{L}_{\text{RL}}.
\]

这说明 MMaDA 的真正 ambition 不只是“提出一个新 backbone”，而是在补齐 diffusion unified model 的全训练栈。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制是“统一 diffusion 预训练 + 统一 diffusion 后训练”的完整范式，而不是某个单独模块。

### 它想解决统一模型里的哪种核心冲突？

它想解决的是统一模型过度以 AR-LM 为中心的范式锁定。也就是：文本为什么必须由 AR 描述、视觉为什么只能是附属分支，这个假设本身被 MMaDA 挑战了。

### 它的关键设计为什么成立？

因为 diffusion 在视觉生成上已经证明了强表达力，作者进一步推断：如果把统一建模对象从“下一个 token”改成“统一扩散过程”，就可能同时覆盖更多任务类型与更连续的生成过程。

### 它相比相邻方法最大的不同点是什么？

和 LLaDA-o 相比，MMaDA 更重视“完整范式”，包括 CoT 与 RL；和 Show-o、Transfusion 相比，它更彻底地把统一中心从语言模型移到 diffusion。

## 与相邻方法的关系

### 它和谁最像？

最像 LLaDA-o，因为两者都属于 diffusion-centered / omni-diffusion unified modeling 家族。

### 它和谁差异最大？

和 Emu3、Chameleon 差异最大，因为后者认为统一应围绕 token-based AR 展开。

### 它继承了什么？

它继承了 diffusion 在视觉生成上的成熟经验，也继承了 unified model 社区对“单一基础框架”的追求。

### 它修正了什么？

它修正的是“统一模型只能以 LLM 为主心骨”的默认设定。

### 它留下了什么问题？

它留下的问题在于：diffusion 作为统一 backbone，在长文本、高频交互和工具使用等方向上，是否能真正建立和 AR-LM 同等成熟的生态。

## 重要细节

- Architecture: 统一 diffusion foundation model
- Objective: diffusion 预训练 + mixed long CoT fine-tuning + UniGRPO 后训练
- Data: 文本推理、多模态理解、图像生成与 CoT 数据
- Evaluation: textual reasoning、multimodal understanding、text-to-image generation
- Strengths: 重新定义统一中心动力学；很重视后训练完整性
- Limitations: diffusion unified model 生态仍新；文本侧效率和交互性仍待进一步验证

## My Take / Why It Matters

MMaDA 在统一模型链条里的意义很像一次范式挑战。它最有价值的地方，不是某项单点性能，而是逼着研究者重新问：统一模型的基础对象到底应该是 token，还是更一般的概率动力学？

它的主要局限是，这条路线还处在系统哲学强于工程成熟度的阶段。但也正因为如此，它对后续 omni-diffusion 方法提供了非常直接的启发。

## 相关笔记

- [LLaDA-o 笔记](./llada-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)

## 参考资料

- Yang et al., "MMaDA: Multimodal Large Diffusion Language Models", arXiv, 2025. https://arxiv.org/abs/2505.15809
- Official repository. https://github.com/Gen-Verse/MMaDA
