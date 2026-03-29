# 统一多模态模型总览

## 元信息

- Topic: unified-models
- Status: seed
- Last updated: 2026-03-29
- Source type: concept
- Primary references:
  - Survey papers and model families spanning joint multimodal representation and generation

## 一句话总结

统一多模态模型试图把多样的输入输出放进更共享的建模框架中，从而减少“每个任务一套系统”的碎片化问题。

## 为什么重要

统一建模有机会减少重复组件、增强跨任务迁移，并把多模态推理重新表述成更一般的序列建模问题。

## 核心思想

### 共享表征

很多统一系统会把不同模态映射到兼容的 token 或 latent 空间中，从而让一个主干网络共同处理。

### 共享目标

如果多种任务能被改写成同一类损失，例如自回归预测或掩码重建，那么训练接口就可以统一。

### 共享接口

当目标是获得广泛能力时，prompt 设计、instruction tuning 和统一的输入输出格式往往和架构本身一样重要。

## 重要细节

- Architecture: 常见形式是 transformer 主干配合模态专属 encoder / decoder
- Objective: 自回归、去噪、对比学习或混合多任务损失
- Data: 混合文本、图像、视频和指令数据
- Evaluation: 迁移广度、多模态推理能力与生成质量
- Strengths: 复用性强、便于迁移、系统叙事更统一
- Limitations: 优化干扰、token 不平衡、评测口径模糊

## 我的笔记

这一页适合作为总览入口，后续随着仓库扩展，再不断拆成更细的模型或方法笔记。

## 开放问题

- 统一到底是在提升泛化，还是只是增加了优化难度？
- 相比主干网络本身，哪些接口设计选择更决定最终效果？

## 相关笔记

- [VLM Reading Map](../vlm/vlm-reading-map.md)

## 参考资料

- 随着这一部分扩展，可以继续补充更具体的模型论文或综述。
