# Janus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-03-29
- Source type: paper
- Primary references:
  - Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
  - Janus official repository

## 一句话总结

Janus 最有价值的判断是：统一模型里的核心冲突不一定出在“是否共享 transformer 主干”，更可能出在“理解和生成是否被迫共用同一种视觉编码”，因此它选择共享认知主干、解耦视觉入口。

## 背景 / 问题设定

很多统一模型默认把视觉统一做到最前端，也就是：

- 一个视觉编码器同时服务理解和生成；
- 同一套视觉表示同时承担判别与重建。

Janus 认为这一步本身就可能有问题：

- 理解需要高层语义、判别性与稳定对齐；
- 生成需要细节保真、可逆性与重建友好表示。

如果二者共用完全相同的视觉编码路径，模型最终可能形成“主干统一了，但视觉表示折中了”的局面。

## 记号

设：

- 理解侧视觉表示为 \(z_u\)
- 生成侧视觉表示为 \(z_g\)
- 共享 transformer 主干为 \(f_\theta\)
- 文本序列为 \(x\)
- 理解目标为 \(\mathcal{L}_{u}\)
- 生成目标为 \(\mathcal{L}_{g}\)

## 核心思想

### 1. 共享主干，不共享视觉编码

Janus 不是完全拆成两个模型，而是在共享 transformer backbone 的前提下，把视觉输入路径解耦。

### 2. 把冲突局部化到视觉表征层

论文的核心判断是：理解和生成的主要矛盾常常出现在视觉表示，而不是语言主干。因此，最值得动刀的位置是视觉入口。

### 3. 统一仍然发生在跨模态上下文建模阶段

尽管前端解耦，Janus 依然坚持一个共享 transformer 处理跨模态上下文，因此它不是“两个模型拼起来”，而是“两个视觉前端 + 一个共享认知核心”。

## 一个简单示意图

```text
image --> understanding encoder --> understanding tokens --+
                                                           |
text ------------------------------------------------------+--> shared AR transformer --> text / multimodal output
                                                           |
prompt --> generation-side visual path --------------------+
```

## Architecture / Data Flow

Janus 的数据流可以概括为“先分后合”。

在理解任务中：

1. 图像进入理解侧视觉编码器；
2. 得到更偏语义与判别的表示 \(z_u\)；
3. \(z_u\) 与文本提示共同进入共享 transformer；
4. 主干输出文本答案或理解相关续写。

在生成任务中：

1. 文本提示和生成侧视觉表示 \(z_g\) 进入统一主干；
2. 主干负责组织跨模态条件；
3. 生成路径根据这些条件完成图像或视觉内容生成。

因此 Janus 不是简单“加两个编码器”，而是把表示流动重新切分：

- 视觉信息在入口处分流；
- 语义与上下文推理在主干处汇合；
- 任务特化在输出端再度分化。

## Training Objective / Recipe

Janus 的训练更适合理解成“联合训练但局部职责清晰”，而不是“一个大 loss 覆盖所有参数”。

总目标可以写成：

\[
\mathcal{L} = \lambda_u \mathcal{L}_{u} + \lambda_g \mathcal{L}_{g}.
\]

但更有信息量的是 recipe：

- 理解样本：图像 + 文本问题，训练共享主干输出文本答案；
- 生成样本：文本提示或多模态提示，训练生成路径输出视觉内容；
- 两类任务共享 transformer，但视觉前端和部分任务专属模块保留独立职责。

论文整体更偏系统设计描述，因此这一节更适合作为基于文中信息的训练理解：

- 共享 transformer 吸收跨模态上下文能力；
- 理解编码器偏向判别性语义；
- 生成编码器偏向可生成视觉表示。

这说明 Janus 的 recipe 创新不在“损失函数发明”，而在“参数职责切分”。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制就是 decoupled visual encoding。论文题眼并不是“统一 transformer”本身，而是“统一 transformer 前面，不必统一视觉入口”。

### 它想解决统一模型里的哪种核心冲突？

它要解决的是：理解需要压缩且判别的视觉表示，而生成需要保真且可逆的视觉表示。这种目标冲突若压在同一视觉编码器上，就容易形成表征折中。

### 它的关键设计为什么成立？

因为共享 transformer 真正负责的是跨模态语义与上下文组织，而不是从像素中提取一套对所有任务都最优的视觉表示。把冲突从主干中拆出去后：

- 主干仍能共享语言与推理能力；
- 理解与生成又能各自保留更合适的视觉表征。

### 它相比相邻方法最大的不同点是什么？

和 Chameleon、Emu3 最大的不同在于，Janus 不相信输入接口应该过度统一；和 DreamLLM 最大的不同在于，它更偏工程拆解而不是能力协同叙事。

## 与相邻方法的关系

### 它和谁最像？

最像 Show-o 和 DreamLLM，因为三者都把“一个模型如何同时理解与生成”当作核心问题。

### 它和谁差异最大？

和 Chameleon、Emu3 这种极端 token 统一路线差异最大，因为 Janus 明确反对在视觉入口层面过度统一。

### 它继承了什么？

它继承了共享 transformer 主干作为统一认知核心的思路，也继承了 AR 多模态建模接口的优点。

### 它修正了什么？

它修正的是“一个视觉编码器打天下”的假设，把统一冲突更准确地定位到视觉侧。

### 它留下了什么问题？

它留下的问题是：随着 unified model 越来越强，视觉编码解耦究竟是长期必要结构，还是一种阶段性工程折中？

## 重要细节

- Architecture: 理解视觉编码器 + 生成视觉编码器 + 共享 AR transformer 主干
- Objective: 多模态理解与生成的联合训练，但视觉入口解耦
- Data: VQA、captioning、图像生成等理解 / 创作混合数据
- Evaluation: multimodal understanding、text-to-image、统一多模态 benchmark
- Strengths: 把冲突定位得很准确；既保留统一主干，又避免视觉入口硬共享
- Limitations: 结构比纯共享更复杂；“统一”形式上没有极端路线那么整齐

## My Take / Why It Matters

Janus 在统一模型谱系里的位置，很像一种“模块化现实主义”校正。它最有价值的地方，不是证明统一做不到，而是证明统一也需要尊重不同任务对表示的不同需求。

它的局限在于，这种解耦策略仍然带有经验性工程判断的色彩，而不是更强理论结论。但正因为它把冲突指出得足够具体，后续很多方法不管继续走极端统一还是混合路线，都不得不回应 Janus 提出的这个问题。

## 相关笔记

- [DreamLLM 笔记](./dreamllm-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)

## 参考资料

- Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation", arXiv, 2024. https://arxiv.org/abs/2410.13848
- Official repository. https://github.com/deepseek-ai/Janus
