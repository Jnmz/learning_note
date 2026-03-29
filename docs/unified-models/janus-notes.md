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

<<<<<<< HEAD
Janus 最重要的判断是：统一模型里的核心冲突不一定出在“是否共享 transformer 主干”，更可能出在“理解和生成是否被迫共用同一种视觉编码”，因此它选择共享认知主干、解耦视觉入口。

## 背景 / 问题设定

很多统一模型默认把视觉统一做到最前端，也就是：

- 一个视觉编码器同时服务理解和生成
- 同一套视觉表示同时承担判别和重建

Janus 认为这一步本身就可能引入冲突：

- 理解需要高层语义、判别性、稳定对齐
- 生成需要细节保真、可逆性、重建友好表示

如果二者共享完全相同的视觉编码路径，模型最终可能“两边都能做，但哪边都不极致”。
=======
Janus 的核心贡献在于指出：统一多模态模型真正的冲突不一定来自“主干是否共享”，而更可能来自“理解和生成是否被迫共用同一种视觉编码”，因此它选择在共享 transformer 主干的同时，解耦视觉编码路径。

## 背景 / 问题设定

很多统一模型在设计上会默认“一个视觉编码器服务所有任务”，但 Janus 认为这一步本身就可能有问题：

- 理解需要高层语义、判别性特征
- 生成需要更适合重建和细节表达的视觉表示

如果二者共用完全相同的视觉编码路径，理解和生成就可能彼此妥协。
>>>>>>> origin/main

## 记号

设：

- 理解侧图像表示为 \(z_u\)
- 生成侧图像表示为 \(z_g\)
- 共享 transformer 主干记为 \(f_\theta\)
- 文本序列为 \(x\)
<<<<<<< HEAD
- 理解目标为 \(\mathcal{L}_{u}\)
- 生成目标为 \(\mathcal{L}_{g}\)
=======
>>>>>>> origin/main

## 核心思想

### 1. 共享主干，不共享视觉编码

<<<<<<< HEAD
Janus 不是把系统拆成两个彼此独立的模型，而是在共享 transformer backbone 的前提下，把视觉输入路径解耦。

### 2. 冲突被局部化到视觉表征层

论文的关键方法判断是：理解和生成的主要矛盾出现在视觉表示，而不是语言主干。因此最值得动刀的位置是视觉入口，而不是整个系统。

### 3. 统一仍然发生在跨模态上下文建模阶段

尽管前端解耦，Janus 依然坚持一个共享 transformer 去处理跨模态上下文，这意味着它不是“理解模型 + 生成模型”组合，而是“两个视觉前端 + 一个共享认知核心”。

## Architecture / Data Flow

Janus 的数据流可以概括为：

```text
image --> understanding encoder --> understanding tokens --+
                                                           |
text ------------------------------------------------------+--> shared AR transformer --> text answer / multimodal continuation
                                                           |
prompt --> generation encoder / generator-side tokens -----+
```

更具体地说：

1. 在理解任务中，图像先进入理解侧视觉编码器，得到更偏语义和判别的表示 \(z_u\)。
2. 在生成任务中，文本提示和生成侧视觉表示 \(z_g\) 进入统一主干，主干负责组织上下文条件。
3. 统一主干之后的输出再交给对应任务路径完成文本生成或图像生成。

关键不在于模块数，而在于信息是“先分后合”：

- 视觉信息在入口处分流
- 语义推理在主干处汇合
- 最终任务在输出端再分化

## Training Objective / Recipe

Janus 的训练理解可以写成一个双任务联合优化：

\[
\mathcal{L} = \lambda_u \mathcal{L}_{u} + \lambda_g \mathcal{L}_{g}.
\]

但这个式子本身信息量不大。更重要的是 recipe：

- 理解样本：图像 + 文本问题，训练共享主干输出文本答案
- 生成样本：文本提示或多模态提示，训练模型输出图像相关生成结果
- 两条任务共享 transformer，但视觉前端和部分任务专属模块保留独立更新空间

因此它更像一种“联合训练但局部参数职责清晰”的方案，而不是全参数强制共享。论文整体更偏系统设计描述，下面这层理解是基于文中结构整理的：

- 共享 transformer 吸收跨模态上下文能力
- 理解编码器偏向保持语义判别性
- 生成编码器偏向保持视觉可生成性

这也解释了为什么 Janus 的 recipe 重点不是损失函数创新，而是参数职责分工。

## Core Mechanism Deep Dive

### 这个方法真正最关键的机制是什么？

最关键的机制是 decoupled visual encoding。Janus 的论文题眼就在这里，它不是泛泛说“加两个视觉模块”，而是明确把冲突定位到视觉编码层。

### 它想解决统一模型里的哪种核心冲突？

它要解决的核心冲突是：理解需要压缩且判别的视觉表示，而生成需要保真且可逆的视觉表示。这种目标冲突如果直接压在同一视觉编码器上，就很容易造成表征折中。

### 它的关键设计为什么成立？

因为共享 transformer 真正负责的是“跨模态语义组织”，而不是“从像素中提取所有任务都通用的最佳视觉表示”。把视觉冲突从主干中拆出去后：

- 主干仍能共享语言和上下文推理能力
- 理解与生成可以各自保留更合适的视觉表征

### 它相比相邻方法最大的不同点是什么？

和 Emu3、Chameleon 最大的不同在于，Janus 不相信输入接口应该过度统一。和 DreamLLM 最大的不同在于，Janus更偏工程拆解而不是能力协同叙事。

## 与相邻方法的关系

### 它和谁最像？

最像 Show-o 和 DreamLLM，因为三者都在认真思考统一模型里的“理解-生成双能力”问题，而不仅是单边任务。

### 它和谁差异最大？

和 Chameleon、Emu3 这类极端 token 统一路线差异最大，因为 Janus 认为统一不该以牺牲视觉表示适配性为代价。

### 它继承了什么？

它继承了共享 transformer 主干作为统一认知核心的设计，也继承了 AR 多模态建模接口的优点。

### 它修正了什么？

它修正的是“一个视觉编码器打天下”的假设，把统一冲突更准确地定位到视觉侧。

### 它留下了什么问题？

它留下的问题是：随着统一模型越来越大，视觉编码解耦到底是长期必要结构，还是过渡期工程折中？如果未来视觉表示本身更强，是否还需要这么明显的解耦？

## 重要细节

- Architecture: 理解视觉编码器 + 生成视觉编码器 + 共享 AR transformer 主干
- Objective: 多模态理解与生成的联合训练，但视觉入口解耦
- Data: VQA、captioning、图像生成等理解 / 创作混合数据
- Evaluation: multimodal understanding、text-to-image、统一多模态 benchmark
- Strengths: 把冲突定位得很准确；既保留统一主干，又避免视觉入口硬共享
- Limitations: 系统结构比纯共享更复杂；“统一”形式上没有极端路线那么整洁

## My Take / Why It Matters

Janus 在统一模型谱系里的位置，很像一种“模块化现实主义”校正。它最有价值的地方，不是证明统一做不到，而是证明统一也需要尊重不同任务对表示的不同需求。

它的局限在于，这种解耦策略仍然是经验性工程判断，而不是更强理论结论。但正因为它把冲突指出得足够具体，后续很多方法不管是继续极端统一，还是转向混合路线，都不得不回应 Janus 提出的这个问题。
=======
Janus 不是完全拆分成两个模型，而是在共享 transformer backbone 的前提下，把理解编码器和生成编码器解耦。

### 2. 视觉冲突主要出现在编码层

论文的核心判断是：理解与生成的矛盾，很多时候不是语言主干不能共享，而是视觉表示要求不同。于是将冲突局部化到视觉侧，是更高性价比的统一路线。

### 3. 用自回归框架统一任务接口

Janus 仍然偏向 autoregressive 统一叙事，通过统一 transformer 来处理跨模态上下文，只是在输入侧给理解和生成不同的视觉表征。

## 关键机制

### Decoupled Visual Encoding

理解路径使用更适合感知和语义抽象的视觉输入表示，生成路径则使用更适合视觉重建和图像生成的表示。这样，模型既保留了统一主干，又避免了单视觉编码器带来的信息粒度冲突。

### Unified Transformer Processing

虽然编码器解耦，但后续跨模态建模仍然在同一个 transformer 中完成，因此 Janus 不是“两个模型拼起来”，而是“两个视觉入口接一个共享认知核心”。

### 扩展性

这种设计的一个好处是，理解和生成两条路径理论上都能独立替换更强组件，而不需要重做整个统一架构。

## 直觉 / 理解

Janus 的价值在于它重新定义了“统一”。它并不执着于从输入到输出每一步都共享，而是更关心哪里应该共享、哪里应该解耦。这个判断比“统一得越彻底越好”更现实。

## 与其他方法的关系

### 对比 Chameleon / Emu3

Chameleon 和 Emu3 更倾向于把视觉和文本都压进统一 token 流。Janus 则认为，视觉表示层面的差异不应被过度抹平。

### 对比 Show-o

Show-o 把冲突处理为“文本 AR、图像离散 diffusion”；Janus 则把冲突处理为“理解视觉编码器和生成视觉编码器解耦”。

### 对比 DreamLLM

DreamLLM 更强调理解-创作协同；Janus 更强调理解-创作冲突的工程化拆解。

## 重要细节

- Architecture: 理解视觉编码器 + 生成视觉编码器 + 共享 transformer 主干
- Objective: 统一多模态 understanding / generation 训练
- Data: 多模态理解数据与图像生成数据
- Evaluation: VQA、captioning、text-to-image、统一多模态 benchmark
- Strengths: 把冲突局部化；兼顾统一与专门化；工程上更灵活
- Limitations: 形式上不如“全共享”统一；两条视觉路径会增加系统复杂度

## 我的笔记 / 开放问题

- Janus 很像统一模型里的“模块化现实主义”。它承认不同任务对视觉表示的需求并不相同。
- 一个值得继续跟踪的问题是：未来更强的 unified model，到底会更接近 Emu3 这种极致统一，还是更接近 Janus 这种共享核心、前端解耦？
>>>>>>> origin/main

## 相关笔记

- [DreamLLM 笔记](./dreamllm-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)

## 参考资料

- Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation", arXiv, 2024. https://arxiv.org/abs/2410.13848
- Official repository. https://github.com/deepseek-ai/Janus
