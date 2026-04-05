# 统一模型

这一部分关注试图在同一个系统中统一多模态、任务类型或交互形式的架构与训练策略。

## 范围

- 共享 token / latent 空间
- 多模态序列建模
- 统一预训练目标
- 理解与生成联合建模

## 笔记

- [统一多模态模型总览](./unified-multimodal-models-overview.md)：梳理统一多模态系统中常见设计模式的起步页。
- [DreamLLM 笔记](./dreamllm-notes.md)：理解 DreamLLM 如何把多模态理解与创作协同作为统一目标来设计。
- [Emu3 笔记](./emu3-notes.md)：分析 Emu3 如何把文本、图像、视频统一到纯 next-token prediction 路线中。
- [Janus 笔记](./janus-notes.md)：分析 Janus 如何通过解耦视觉编码来统一理解和生成。
- [Transfusion 笔记](./transfusion-notes.md)：分析 Transfusion 如何在一个主干里组合语言建模和图像 diffusion。
- [Orthus 笔记](./orthus-notes.md)：分析 Orthus 如何在自回归主干下结合语言 head 与 diffusion head。
- [Chameleon 笔记](./chameleon-notes.md)：分析 Chameleon 的 mixed-modal early-fusion token 路线。
- [MMaDA 笔记](./mmada-notes.md)：分析 MMaDA 如何把统一模型推进到 diffusion foundation model 路线。
- [LLaDA-o 笔记](./llada-o-notes.md)：分析 LLaDA-o 的 Mixture of Diffusion 与长度自适应设计。
- [Uni-RS 笔记](./uni-rs-notes.md)：分析面向遥感领域的空间忠实统一理解与生成模型。
- [Show-o 笔记](./show-o-notes.md)：详细分析 Show-o 如何在一个 transformer 里结合语言自回归建模与离散扩散式图像生成。
- [Show-o2 笔记](./show-o2-notes.md)：详细分析 Show-o2 如何用 3D causal VAE latent 与 flow matching 进一步统一文本、图像与视频。
- [TUNA 笔记](./tuna-notes.md)：详细分析 TUNA 如何通过 VAE latent 与 representation encoder 级联，构造统一连续视觉表示。
- [InternVL-U 笔记](./internvl-u-notes.md)：详细分析 InternVL-U 如何以统一上下文建模结合 MMDiT generation head，同时覆盖理解、推理、生成与编辑。

## 后续可补充的笔记

- token 统一策略
- 模态适配器与路由机制
- 跨任务训练配比
- all-in-one 系统中的评测权衡
