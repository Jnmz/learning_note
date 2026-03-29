# VLM 阅读地图

## 元信息

- Topic: vlm
- Status: seed
- Last updated: 2026-03-29
- Source type: concept
- Primary references:
  - Foundational VLM pretraining and instruction-tuning papers

## 一句话总结

视觉语言模型可以从三个角度来组织理解：视觉与文本如何对齐、信息在什么位置融合，以及训练依赖了什么监督信号。

## 为什么重要

VLM 文献增长很快，用清晰的分类框架来读论文，更容易比较设计选择，同时不丢掉整体脉络。

## 核心思想

### 对齐策略

有些系统依赖对比式对齐，有些采用生成式训练，也有些主要借助 instruction-following 监督。

### 融合位置

不同模型会在早期融合、后期融合，或者通过轻量 connector 接入预训练语言模型。

### 能力划分

阅读新的 VLM 论文时，把偏感知的任务和偏推理的任务分开看，往往会更清楚。

## 重要细节

- Architecture: 视觉编码器配合文本编码器或 LLM 主干
- Objective: 对比学习、captioning、next-token prediction 或 instruction tuning
- Data: 图文对、问答数据、OCR 密集语料与合成指令数据
- Evaluation: captioning、VQA、grounding、文档任务与 agent 式推理
- Strengths: 监督丰富，接口实用，多模态交互自然
- Limitations: grounding 脆弱、容易 hallucination、对 benchmark 选择敏感

## 我的笔记

这一页适合作为后续扩展 CLIP 类、Flamingo 类和 LLM-connector 类 VLM 笔记的枢纽页。

## 开放问题

- 现在还有哪些 benchmark 真正在衡量多模态推理，而不是语言先验？
- 最近的进展中，数据整理的贡献和架构创新的贡献各占多少？

## 相关笔记

- [Unified Multimodal Models Overview](../unified-models/unified-multimodal-models-overview.md)

## 参考资料

- 随着这一部分扩展，可以在这里逐步补充具体论文笔记。
