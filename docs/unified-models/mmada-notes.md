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

MMaDA 代表了一条与 AR 统一路线明显不同的方向：它试图把文本推理、多模态理解和图像生成统一到一个 diffusion foundation model 中，并进一步配套统一的 CoT 微调与 diffusion 专用强化学习。

## 背景 / 问题设定

到 2025 年前后，统一模型大多仍以自回归 transformer 为默认核心，而 MMaDA 提出一个更激进的问题：

- 如果 diffusion 在视觉生成上已经很强，是否也能成为更一般的多模态 foundation model 框架？
- 统一不一定要围绕 next-token prediction 展开，是否可以围绕统一 diffusion 概率形式展开？

## 记号

设：

- 多模态 token / state 序列为 \(x\)
- 扩散时间步为 \(t\)
- 模型参数为 \(\theta\)
- 统一 diffusion 目标记为 \(\mathcal{L}_{\text{diff}}\)
- 强化学习阶段目标记为 \(\mathcal{L}_{\text{RL}}\)

## 核心思想

### 1. 统一 diffusion 架构

MMaDA 不把 diffusion 当成单独的图像生成组件，而是把它提升为统一 foundation model 的基本建模形式。

### 2. 模态无关设计

论文特别强调 modality-agnostic design，目标是尽量减少模态专属子模块，让文本、图像和多模态任务能在同一概率框架下被处理。

### 3. 从预训练一路打通到后训练

MMaDA 不只提预训练架构，还把 mixed long CoT fine-tuning 和 UniGRPO 强化学习一起纳入统一方法论，试图把 reasoning 与 generation 的后训练也统一起来。

## 关键机制

### Unified Diffusion Pretraining

模型在统一 diffusion 目标下学习多模态基础能力，这与传统“文本 AR + 图像 diffusion”的混合路线不同。

### Mixed Long CoT Fine-Tuning

这部分很有辨识度。论文认为跨模态 reasoning 不应该只停留在结构统一，还要把 textual CoT 与 visual / multimodal CoT 的格式与习惯统一起来。

### UniGRPO

UniGRPO 是针对 diffusion foundation model 设计的统一 RL 算法。它说明 MMaDA 不只是一个预训练架构提案，而是在试图建立“diffusion 版统一大模型”的完整训练范式。

## 直觉 / 理解

MMaDA 像是在挑战一个默认假设：统一模型未必要围绕 LLM 范式组织，diffusion 也可能成为更一般的认知与生成基础框架。

## 与其他方法的关系

### 对比 Emu3 / Chameleon

Emu3 和 Chameleon 强调 AR token 统一；MMaDA 则把统一建立在 diffusion 概率建模上。

### 对比 Show-o / Transfusion

Show-o 和 Transfusion 都是“文本侧仍偏 AR”的折中路线；MMaDA 则更彻底地把统一中心移向 diffusion。

### 对比 LLaDA-o

两者都属于 omni-diffusion 叙事，但 LLaDA-o 更强调混合离散 / 连续 diffusion 与长度自适应，MMaDA 更强调统一后训练与 RL。

## 重要细节

- Architecture: 统一 diffusion foundation model
- Objective: 多模态 diffusion 预训练 + mixed long CoT 微调 + UniGRPO 后训练
- Data: 文本推理、多模态理解、图像生成数据
- Evaluation: textual reasoning、multimodal understanding、text-to-image generation
- Strengths: 把统一从预训练延展到后训练；扩散路线更原教旨
- Limitations: diffusion 统一范式仍较新；文本侧效率和推理习惯仍需更多验证

## 我的笔记 / 开放问题

- MMaDA 的重要性在于它不再接受“语言必须 AR、视觉才 diffusion”这个分工，而是重新提问统一模型的中心动力学。
- 这条路线能否在大规模长上下文文本任务上真正和 AR LLM 抗衡，是接下来最关键的问题之一。

## 相关笔记

- [LLaDA-o 笔记](./llada-o-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)
- [Show-o2 笔记](./show-o2-notes.md)

## 参考资料

- Yang et al., "MMaDA: Multimodal Large Diffusion Language Models", arXiv, 2025. https://arxiv.org/abs/2505.15809
- Official repository. https://github.com/Gen-Verse/MMaDA
