# 统一模型

这一部分关注试图在同一个系统中统一多模态、任务类型或交互形式的架构与训练策略。

## 范围

- 共享 token / latent 空间
- 多模态序列建模
- 统一预训练目标
- 理解与生成联合建模

## 笔记

- [统一多模态模型总览](./unified-multimodal-models-overview.md)：梳理统一多模态系统中常见设计模式的起步页。
- [Show-o 笔记](./show-o-notes.md)：详细分析 Show-o 如何在一个 transformer 里结合语言自回归建模与离散扩散式图像生成。
- [Show-o2 笔记](./show-o2-notes.md)：详细分析 Show-o2 如何用 3D causal VAE latent 与 flow matching 进一步统一文本、图像与视频。

## 后续可补充的笔记

- token 统一策略
- 模态适配器与路由机制
- 跨任务训练配比
- all-in-one 系统中的评测权衡
