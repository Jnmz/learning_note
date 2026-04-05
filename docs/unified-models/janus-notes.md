# Janus 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
  - Janus official repository

## 一句话总结

Janus 最有价值的判断是：统一模型里的核心冲突不一定出在“是否共享 transformer 主干”，更可能出在“理解和生成是否被迫共用同一种视觉编码”；因此它选择共享认知主干、解耦视觉入口，把理解侧和生成侧的视觉表征需求分开处理。

## 背景 / 问题设定

很多 unified multimodal model 默认会把“统一”一直推到视觉入口，也就是：

- 一个视觉编码器同时服务理解和生成；
- 同一套视觉表示同时承担判别与重建；
- 主干看到的视觉对象在理解和创作两边尽量保持一致。

Janus 认为，这一步本身就可能有问题。因为理解和生成对视觉表示的偏好并不相同：

1. 理解更需要高层语义、判别性、稳定对齐，以及有利于回答问题和推理的抽象表示。
2. 生成更需要细节保真、可逆性、局部结构可恢复，以及适合 token 化或重建的表示。

如果二者被迫共享完全相同的视觉编码路径，就容易出现一个很典型的折中：

\[
\text{shared backbone succeeded},
\qquad
\text{visual representation compromised}.
\]

也就是说，模型表面统一了，但视觉入口变成了两种任务之间的妥协点。

因此 Janus 的问题设定非常精准：

- 真正值得共享的，究竟是视觉前端、语言主干，还是二者都共享？
- 如果统一模型的主要冲突在视觉表征层，那么最合理的改法是否应当是“局部解耦，而不是整体拆开”？

这使 Janus 在 unified model 谱系里很特别。它不像 Chameleon/Emu3 那样极端追求接口统一，也不像完全外挂式系统那样彻底拆成两套模型，而是在“共享主干、解耦入口”这个位置上给出了一种很现实的结构修正。

## 记号

设：

- 输入图像为 \(I\)
- 理解侧视觉编码器为 \(E_u\)
- 生成侧视觉编码器或视觉 tokenizer 为 \(E_g\)
- 理解侧视觉表示为 \(z_u = E_u(I)\)
- 生成侧视觉表示为 \(z_g = E_g(I)\)
- 文本序列为 \(x = (x_1,\dots,x_T)\)
- 共享 transformer 主干为 \(f_\theta\)
- 理解目标为 \(\mathcal{L}_u\)
- 生成目标为 \(\mathcal{L}_g\)

Janus 的核心结构关系可以抽象写成

\[
z_u = E_u(I), \qquad z_g = E_g(I),
\]

以及

\[
h = f_\theta(\text{text context}, z_u \text{ or } z_g).
\]

也就是说，视觉信息在入口处分流，但跨模态语义和上下文推理在共享主干处重新汇合。

## 核心思想

### 1. 共享主干，不共享视觉编码

Janus 不是完全拆成两个模型，而是在共享 transformer backbone 的前提下，把视觉输入路径解耦。

这一点非常关键，因为它说明 Janus 的目标不是“否定 unified model”，而是“更精细地定义应该统一到哪里”。

### 2. 把冲突局部化到视觉表征层

论文的核心判断是：理解和生成的主要矛盾常常出现在视觉表示，而不是语言主干。

换句话说，Janus 认为问题不一定是

\[
\text{shared transformer is bad},
\]

而更可能是

\[
\text{forcing one visual encoder to satisfy two incompatible objectives is bad}.
\]

因此，最值得动刀的位置是视觉入口，而不是整个统一架构。

### 3. 统一仍然发生在跨模态上下文建模阶段

尽管前端解耦，Janus 依然坚持一个共享 transformer 处理跨模态上下文，因此它不是“两个模型拼起来”，而是：

\[
\text{two visual front-ends}
\; + \;
\text{one shared cognitive core}.
\]

这使它仍然属于 unified model，只是它统一的不是视觉编码器，而是语义组织和推理主干。

## 一个简单示意图

```text
image --> understanding encoder --> understanding tokens ----+
                                                             |
text --------------------------------------------------------+--> shared AR transformer --> text / multimodal output
                                                             |
prompt / image --> generation-side visual tokenizer ---------+
```

## 详细推导

### 推导 1：若共用同一视觉编码器，理解与生成目标会在表示层发生冲突

假设存在单一视觉编码器 \(E\)，同时服务理解和生成，那么视觉表示只有一套：

\[
z = E(I).
\]

此时模型的总目标可以写成

\[
\mathcal{L}(E, \theta)
=
\lambda_u \mathcal{L}_u(E, \theta)
+
\lambda_g \mathcal{L}_g(E, \theta).
\]

对共享视觉编码器参数求梯度，得到

\[
\nabla_E \mathcal{L}
=
\lambda_u \nabla_E \mathcal{L}_u
+
\lambda_g \nabla_E \mathcal{L}_g.
\]

如果 \(\mathcal{L}_u\) 偏好判别性语义压缩，而 \(\mathcal{L}_g\) 偏好可重建细节保真，那么这两项梯度很可能在表示层方向不一致。于是共享编码器学到的不是最适合理解的表示，也不是最适合生成的表示，而是一个折中表示。

这正是 Janus 想指出的矛盾来源：冲突不一定在共享主干，而可能更早就发生在视觉入口。

### 推导 2：视觉解耦相当于把冲突从同一个参数子空间中拆开

Janus 改成使用两套视觉路径后，表示变成

\[
z_u = E_u(I), \qquad z_g = E_g(I).
\]

于是总目标变成

\[
\mathcal{L}(E_u, E_g, \theta)
=
\lambda_u \mathcal{L}_u(E_u, \theta)
+
\lambda_g \mathcal{L}_g(E_g, \theta).
\]

此时对两套编码器的梯度分别是

\[
\nabla_{E_u} \mathcal{L} = \lambda_u \nabla_{E_u} \mathcal{L}_u,
\qquad
\nabla_{E_g} \mathcal{L} = \lambda_g \nabla_{E_g} \mathcal{L}_g.
\]

也就是说，理解和生成不再在同一个视觉参数子空间里直接打架。Janus 的核心收益并不神秘，本质上是把冲突局部化并显式拆开。

这就是为什么它看上去只是“加一个编码器”，但方法意义其实很大：它改变了冲突发生的位置。

### 推导 3：共享主干仍然在学习统一的跨模态条件建模

虽然视觉入口解耦，但共享主干仍然要处理理解和生成两类任务。设共享主干输出为

\[
h = f_\theta(x, z_u) \quad \text{or} \quad h = f_\theta(x, z_g).
\]

对理解任务，主干学习的是

\[
p_\theta(\text{answer} \mid x, z_u),
\]

对生成任务，主干学习的是

\[
p_\theta(\text{visual tokens or generation condition} \mid x, z_g).
\]

这意味着 Janus 统一的不是视觉表征本身，而是条件组织能力：

\[
\text{shared transformer}
\approx
\text{shared cross-modal reasoning core}.
\]

因此它不像“完全分家”的两模型系统，因为理解和生成仍然在同一个语义主干里交换归纳偏置。

### 推导 4：Janus 的总目标体现的是“入口解耦，认知共享”

Janus 的联合训练目标可以写成

\[
\mathcal{L}
=
\lambda_u \mathcal{L}_u
+
\lambda_g \mathcal{L}_g.
\]

这个式子表面上普通，但真正有信息量的是参数依赖结构：

\[
\mathcal{L}_u = \mathcal{L}_u(E_u, f_\theta),
\qquad
\mathcal{L}_g = \mathcal{L}_g(E_g, f_\theta).
\]

这里 \(f_\theta\) 是共享的，而 \(E_u, E_g\) 是分开的。于是 Janus 其实在明确做一种结构化分工：

- 视觉输入分开学；
- 跨模态语义一起学；
- 输出再按任务类型分化。

这和 Chameleon/Emu3 那种“入口也统一”的路线形成鲜明对比，也和完全双模型系统不同，因为认知主干仍然是同一个。

## 架构理解

### 1. 为什么 Janus 的关键不只是“两个编码器”

如果只是把两个视觉编码器拼上去，但后面还是两个独立系统，那就没有统一可言。Janus 的关键在于：

- 前端视觉路径解耦；
- 共享 transformer 继续承担统一语义主干；
- 模型仍在一个认知空间里组织理解与生成的上下文。

所以它不是在“多堆模块”，而是在更精确地划分什么该共享、什么不该共享。

### 2. 为什么它把问题定位得很准

Janus 的价值在于它对冲突定位很具体。很多方法只说“理解和生成可能冲突”，但 Janus 进一步说：

\[
\text{the bottleneck may be the visual encoding interface}.
\]

这让问题从抽象的“多任务冲突”变成了一个可操作的结构设计问题。

### 3. 为什么它仍然属于 unified model，而不是 modular ensemble

虽然视觉入口解耦，Janus 仍然不是“理解模型 + 生成模型”的外部拼接，因为：

- 它们共享一个 transformer 主干；
- 共享同一套跨模态语言与推理能力；
- 统一上下文建模仍然发生在主干内部。

所以 Janus 更像一种“局部解耦的统一模型”，而不是彻底分家。

## 训练流程

根据论文和公开材料，Janus 的训练重点不是某个复杂新 loss，而是如何让两套视觉前端与共享主干协同工作。

### 阶段 1：理解侧视觉路径对齐语义任务

先让理解侧视觉编码器稳定服务 VQA、caption、图像理解等任务，学习更偏判别和语义化的视觉表示。

### 阶段 2：生成侧视觉路径对齐创作任务

再让生成侧视觉路径服务图像生成或相关创作任务，学习更适合重建、token 化或生成条件的表示。

### 阶段 3：共享主干吸收两类上下文建模信号

在两套视觉路径各司其职的基础上，统一主干持续从两类任务中学习跨模态上下文组织能力。这样一来：

- 理解侧不会为了生成而被迫保留多余细节；
- 生成侧也不会为了判别性而丢掉可重建性；
- 主干则尽量保住统一的认知与语言推理能力。

## 直觉 / 理解

我对 Janus 的理解是：它像 unified model 领域里一次很重要的“定位校正”。很多人看到 unified model，就会下意识地把统一往所有层次都推到底；但 Janus 说，不一定。统一也可以是有边界的，关键是找到真正值得共享的层。

它的美感不在于形式极简，而在于结构判断很准：理解和生成共享一个认知核心没有问题，但让它们共享完全同一套视觉入口，可能本来就是错位的要求。

## 与相邻方法的关系

### 对比 Chameleon / Emu3

Chameleon、Emu3 相信输入接口应该尽量统一，最好一路统一到 token 流和目标函数。Janus 则明确站在另一端：它认为视觉入口不应过度统一，否则理解和生成会在表示层过早发生冲突。

### 对比 DreamLLM

DreamLLM 更强调 comprehension-creation synergy 和共享语义主干，但对冲突位置的刻画没 Janus 这么具体。Janus 更像是在回答：如果 synergy 不能靠硬共享入口自然出现，那具体应该在哪一层做解耦？

### 对比 Show-o

Show-o 在统一主干中组合文本 AR 和视觉离散去噪，重点是统一 backbone 下的 mixed objective。Janus 则更早一步，把焦点放在视觉入口本身是否应该统一。

### 对比 Orthus / Transfusion

Orthus、Transfusion 都体现“共享主干 + 模态专属输出动力学”的思路。Janus 的独特之处在于，它把“专属化”更明确地前移到了视觉编码入口，而不只是输出端。

## 重要细节

- Architecture: 理解视觉编码器 + 生成视觉编码器 / tokenizer + 共享 AR transformer 主干
- Objective: 多模态理解与生成联合训练，但视觉入口解耦
- Representation: 理解侧偏语义判别，生成侧偏可生成 / 可重建表示
- Data: VQA、caption、图像生成等理解 / 创作混合数据
- Evaluation: multimodal understanding、text-to-image、统一多模态 benchmark
- Strengths: 把冲突定位得很准确；既保留统一主干，又避免视觉入口硬共享
- Limitations: 结构比纯共享更复杂；统一形式上不如极端路线整齐；解耦边界仍带有工程经验色彩

## 我的笔记 / 开放问题

### 我的笔记

我觉得 Janus 在 unified model 谱系里的意义很大，因为它把一个常被模糊讨论的问题讲具体了：到底是 unified model 本身有问题，还是我们把不该共享的部分也强行共享了？Janus 的回答是，问题往往不在共享主干，而在共享视觉入口。

这使它很像一种“模块化现实主义”校正。它没有否定 unified model，只是把统一做得更有边界感。

### 开放问题

- 随着视觉 tokenizer 和生成表示变强，视觉入口解耦会一直必要，还是只是过渡阶段的工程折中？
- 理解侧和生成侧编码器之间是否还能进一步共享一些低层参数，而不重新引入冲突？
- 当统一模型规模继续增大时，共享主干会不会反过来要求视觉入口再次靠拢？
- 如果视频也加入进来，Janus 这种“入口解耦”会不会需要继续扩成三路甚至更多路？

## 相关笔记

- [DreamLLM 笔记](./dreamllm-notes.md)
- [Show-o 笔记](./show-o-notes.md)
- [Chameleon 笔记](./chameleon-notes.md)
- [Transfusion 笔记](./transfusion-notes.md)

## 参考资料

- Wu et al., "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation", arXiv, 2024. https://arxiv.org/abs/2410.13848
- Official repository. https://github.com/deepseek-ai/Janus
- Hugging Face paper page. https://huggingface.co/papers/2410.13848

