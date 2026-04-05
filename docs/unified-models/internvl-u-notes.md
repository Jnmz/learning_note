# InternVL-U 笔记

## 元信息

- Topic: unified-models
- Status: revised
- Last updated: 2026-04-06
- Source type: paper
- Primary references:
  - InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing
  - InternVL-U official repository

## 一句话总结

InternVL-U 的关键不是把所有模态硬塞进同一种输出形式，而是把“统一”放在上下文建模层，把“专门化”留在视觉生成动力学层：前者沿用强 MLLM 的语义与推理能力，后者接一个专门的 MMDiT generation head，再通过高语义密度的数据合成，把理解、推理、生成、编辑真的拉进同一个系统里。

## 背景 / 问题设定

统一多模态模型（Unified Multimodal Model, UMM）在过去一段时间里遇到的核心矛盾非常稳定：

1. 如果系统更像 MLLM，那么理解、问答、推理通常更强，但视觉生成和编辑容易偏弱。
2. 如果系统更像 diffusion / flow generation model，那么图像质量和编辑能力会更强，但高层语义理解、指令跟随和推理又容易下降。
3. 如果强行共享一切，包括视觉表示、主干和输出头，那么训练时常会出现目标冲突，最后谁也不够强。

InternVL-U 要解决的并不是“如何用更大参数 brute force 统一一切”，而是一个更克制的问题：

- 能不能在只有 4B 参数量级时，同时保住理解 / 推理能力和生成 / 编辑能力？
- 统一模型的真正共享位置，究竟应当是 token 接口、视觉编码器、还是更高层的上下文语义建模？
- 如果高层语义和低层视觉细节天然偏好不同模块，统一模型是否应该接受“主干统一 + 头部专门化”？

论文给出的答案很明确：

\[
\text{unified contextual modeling}
\;+\;
\text{modality-specific modular design}
\]

再配合

\[
\text{decoupled visual representations}
\]

来实现统一能力，而不是坚持“一个头解决全部动力学”。

## 记号

设：

- 输入文本指令为 \(x\)
- 输入图像为 \(I\)
- 理解侧视觉编码器输出的视觉 token 为 \(v_u\)
- 生成 / 编辑侧视觉 latent 或条件表示为 \(v_g\)
- 共享 MLLM 主干为 \(f_\theta\)
- 主干输出的统一上下文表示为 \(h = f_\theta(x, v_u)\)
- MMDiT generation head 为 \(g_\phi\)
- 真实目标图像经 VAE 编码后的 latent 为 \(z_1\)
- 噪声 latent 为 \(z_0 \sim \mathcal{N}(0, I)\)
- 插值时刻为 \(t \in [0,1]\)
- 插值路径上的 noisy latent 为 \(z_t\)
- 条件上下文为 \(c\)，它由文本、图像、历史输出等统一组织得到

于是 InternVL-U 的基本结构可抽象写成：

\[
h = f_\theta(x, v_u),
\qquad
\hat{v} = g_\phi(z_t, t, c(h, v_g)).
\]

其中 \(f_\theta\) 负责语义组织、指令理解、推理和条件整合，\(g_\phi\) 负责视觉生成或编辑所需的连续动力学建模。

## 核心思想

### 1. 统一发生在 context，而不是输出动力学

InternVL-U 最值得注意的地方，是它并不追求“文本和图像都用同一种 next-token prediction”这种极端统一，也不追求“一个视觉头同时做理解与生成”的极端共享。

它的统一点是：

- 所有任务都先组织到同一个多模态上下文里；
- 统一主干负责读懂文本、图像、推理链和编辑意图；
- 真正需要连续视觉建模时，再调用专门的 MMDiT generation head。

也就是说，InternVL-U 更相信的是：

\[
\text{share semantics first, specialize dynamics later}.
\]

### 2. 理解侧与生成侧的视觉表征解耦

论文明确强调 decoupled visual representations。这个判断与 Janus 有相似处，但 InternVL-U 的重点更偏“如何把一个强 MLLM 和一个强生成头拼成真正能协作的统一系统”。

理解侧视觉表示更偏：

- 高层语义
- 对齐语言
- 支撑推理

生成 / 编辑侧视觉表示更偏：

- 局部结构
- 纹理与文字细节
- 可逆或易于连续建模的 latent

如果强行共用同一种视觉表示，最常见的问题就是：

\[
\text{semantic abstraction}
\quad \text{vs.} \quad
\text{pixel / layout faithfulness}
\]

在同一表示空间里互相拉扯。

### 3. 高语义密度数据是统一能力成立的关键

InternVL-U 不是只靠架构取胜。论文很强调 reasoning-centric data synthesis，尤其关心那些“仅靠审美好看还不够”的任务：

- text rendering
- scientific reasoning
- text-aware image editing
- compositional image editing

这背后有一个很重要的判断：统一模型若想真正覆盖理解、推理、生成、编辑，训练数据就不能只偏 aesthetic generation，而必须显式提供高层语义约束和中间推理结构。

## 一个简单示意图

```text
image/text prompt ------------------------------+
                                                |
image for understanding --> visual encoder -----+--> shared MLLM --> unified context states
                                                |                        |
image for editing / target conditions ----------+                        |
                                                                         +--> text / reasoning output
                                                                         |
noisy latent z_t + timestep t + generation conditions -------------------+--> MMDiT generation head --> image / edited image
```

## 详细推导

### 推导 1：InternVL-U 的统一，本质上是统一条件上下文建模

无论是多模态理解、视觉推理、文生图还是图像编辑，InternVL-U 都先把任务写成“给定统一上下文 \(c\)，预测目标输出”的问题。

若输出是文本答案 \(y = (y_1,\dots,y_T)\)，那么就是标准条件语言建模：

\[
p_\theta(y \mid c)
=
\prod_{i=1}^{T} p_\theta(y_i \mid y_{<i}, c).
\]

对应的负对数似然损失为

\[
\mathcal{L}_{\text{text}}
=
- \sum_{i=1}^{T}\log p_\theta(y_i \mid y_{<i}, c).
\]

这里的 \(c\) 可以包含：

- 文本 instruction
- 输入图像的理解 token
- 历史对话
- 中间推理链（Chain-of-Thought, CoT）

如果任务是图像生成或图像编辑，InternVL-U 不再直接预测像素，也不把图像离散成 token 做纯 AR，而是交给 generation head 在连续 latent 空间里建模。因此统一的关键不是“文本和图像都用同一种 loss”，而是：

\[
\text{all tasks share the same semantic conditioning pipeline}.
\]

这就是论文里 unified contextual modeling 的真正含义。

### 推导 2：为什么“解耦视觉表示”能够减少理解与生成的梯度冲突

先看反例。若理解与生成共用同一视觉表示 \(v = E(I)\)，总目标可以写成

\[
\mathcal{L}(E, \theta, \phi)
=
\lambda_u \mathcal{L}_{\text{understand}}(E, \theta)
+
\lambda_g \mathcal{L}_{\text{generate}}(E, \theta, \phi).
\]

对共享视觉编码器 \(E\) 求梯度，有

\[
\nabla_E \mathcal{L}
=
\lambda_u \nabla_E \mathcal{L}_{\text{understand}}
+
\lambda_g \nabla_E \mathcal{L}_{\text{generate}}.
\]

如果理解目标偏好抽象语义压缩，而生成目标偏好精细可恢复结构，那么两项梯度方向往往不一致。于是共享表示 \(v\) 容易退化成折中结果：

\[
v \approx \text{neither best for reasoning nor best for synthesis}.
\]

InternVL-U 改成两套视觉路径后：

\[
v_u = E_u(I), \qquad v_g = E_g(I),
\]

总目标变成

\[
\mathcal{L}
=
\lambda_u \mathcal{L}_{\text{understand}}(E_u, \theta)
+
\lambda_g \mathcal{L}_{\text{generate}}(E_g, \theta, \phi).
\]

此时梯度拆为

\[
\nabla_{E_u}\mathcal{L}
=
\lambda_u \nabla_{E_u}\mathcal{L}_{\text{understand}},
\qquad
\nabla_{E_g}\mathcal{L}
=
\lambda_g \nabla_{E_g}\mathcal{L}_{\text{generate}}.
\]

也就是说，视觉冲突不再直接发生在同一个参数子空间里。共享部分被后移到了 MLLM 的上下文建模阶段，因此模型的结构性判断变成：

\[
\text{decouple representation conflict,}
\quad
\text{retain semantic sharing}.
\]

### 推导 3：MMDiT generation head 可以写成标准条件 flow matching

论文说明视觉生成头基于 MMDiT。虽然技术报告中的完整实现细节比摘要更丰富，但从统一建模角度看，可以把它理解为条件 flow / diffusion 系列方法中的一类连续 latent 预测器。

设真实图像 latent 为 \(z_1\)，噪声为 \(z_0 \sim \mathcal{N}(0,I)\)。采用最常见的线性插值路径：

\[
z_t = (1-t)z_0 + t z_1.
\]

对 \(t\) 求导，得到目标速度场

\[
\frac{d z_t}{dt} = z_1 - z_0.
\]

若 generation head 预测速度

\[
v_{\theta,\phi}(z_t, t, c),
\]

则标准 flow matching 目标可写为

\[
\mathcal{L}_{\text{flow}}
=
\mathbb{E}_{z_0, z_1, t}
\left[
\left\|
v_{\theta,\phi}(z_t, t, c) - (z_1 - z_0)
\right\|_2^2
\right].
\]

这个式子里的关键不是公式本身，而是条件 \(c\) 的来源。InternVL-U 中的 \(c\) 不是一个简单文本 embedding，而是来自统一 MLLM 语义主干整合后的上下文，因此 generation head 学到的是：

\[
\text{visual dynamics conditioned on reasoning-aware multimodal context}.
\]

这和普通文生图模型相比，多了一层统一系统语义调度；和普通 MLLM 相比，则多了真正强的连续视觉生成能力。

### 推导 4：图像编辑只是带视觉条件约束的条件生成

图像编辑任务可以形式化为：给定源图像 \(I_s\)、编辑指令 \(x\)，生成目标图像 \(I_t\)。如果把源图像的编辑条件编码为 \(e(I_s)\)，把文本指令编码进统一上下文 \(c\)，那么编辑目标可以写成

\[
p(I_t \mid I_s, x)
\approx
p(z_t^{\text{target}} \mid e(I_s), c).
\]

在 flow matching 记号下，就是让 generation head 学会

\[
v_{\theta,\phi}(z_t, t, c, e(I_s)).
\]

因此编辑并不需要引入另一套根本不同的公式；它只是比文生图多了一个“必须保留哪些内容、必须修改哪些内容”的条件约束。

如果进一步把编辑要求拆成“保留项”和“修改项”，则可以抽象写成

\[
I_t
=
\operatorname{Edit}(I_s; \text{preserve}, \text{modify}).
\]

多代理数据合成流程的价值，恰好在于把这种结构化编辑意图显式做出来，例如：

- Global：全局风格或整体场景变化
- Object：对象增删替换
- Attribute：颜色、材质、大小等属性变换
- Compositional：多约束联合编辑

这使模型学习到的不是“模糊地改一张图”，而是更结构化的条件编辑算子。

### 推导 5：CoT 数据为什么会帮助生成与编辑，而不仅仅帮助问答

这一点很容易被忽略。论文强调 reasoning-centric synthesis，并不是只为了让模型在 MMMU 一类 benchmark 上分数更高，而是为了把高层抽象意图更稳定地映射到视觉细节决策。

设用户意图为 \(u\)，最终视觉目标为 \(I\)。如果不显式建模中间推理链 \(r\)，系统往往直接学习

\[
p(I \mid u).
\]

但当任务语义密度很高时，例如科学绘图、复杂文本渲染、具备空间约束的构图，这个分布非常宽，优化难度大，因为从抽象意图到视觉细节之间缺少中间结构。

若引入 CoT 样式中间推理 \(r\)，则可写成

\[
p(I \mid u)
=
\sum_r p(I \mid r, u)\, p(r \mid u).
\]

这就是条件概率分解。它说明：如果模型能先形成较好的中间推理表示 \(r\)，再据此决定视觉布局、文字位置、对象关系，那么最终的视觉生成条件分布会更集中、更可学。

虽然训练时未必显式枚举所有 \(r\)，但 reasoning-centric 数据实际上是在逼近这样一种因子化：

\[
u \rightarrow r \rightarrow I.
\]

这正是 InternVL-U 把 CoT 引入高语义密度生成任务的理论动机。

### 推导 6：联合目标体现的是“共享认知主干 + 专门视觉动力学”

综合起来，InternVL-U 的训练可抽象写成

\[
\mathcal{L}
=
\lambda_{\text{text}} \mathcal{L}_{\text{text}}
+
\lambda_{\text{flow}} \mathcal{L}_{\text{flow}}
+
\lambda_{\text{edit}} \mathcal{L}_{\text{edit}}
+
\lambda_{\text{aux}} \mathcal{L}_{\text{aux}}.
\]

其中：

- \(\mathcal{L}_{\text{text}}\) 负责理解与推理输出
- \(\mathcal{L}_{\text{flow}}\) 负责连续视觉生成
- \(\mathcal{L}_{\text{edit}}\) 可视为带源图条件的视觉生成项
- \(\mathcal{L}_{\text{aux}}\) 代表对齐、投影器或其他辅助项

真正重要的是参数依赖结构，而不是“loss 相加”本身：

\[
\mathcal{L}_{\text{text}} = \mathcal{L}_{\text{text}}(\theta, E_u),
\]

\[
\mathcal{L}_{\text{flow}} = \mathcal{L}_{\text{flow}}(\theta, \phi, E_g),
\]

\[
\mathcal{L}_{\text{edit}} = \mathcal{L}_{\text{edit}}(\theta, \phi, E_g).
\]

这说明统一点主要在 \(\theta\) 上，也就是统一语义上下文建模；而视觉生成与编辑能力主要通过 \(\phi\) 和 \(E_g\) 承担。这种参数分工就是 InternVL-U 的设计哲学。

## 架构理解

### 1. 为什么它本质上是 “MLLM + generation head” 而不是简单拼接

如果只是把一个 MLLM 和一个文生图模型拼在一起，通常只能做到“理解后调用生成器”，但很难在端到端训练中真正共享条件表示。

InternVL-U 更进一步的地方在于：

- 共享主干负责统一组织文本、图像和推理信息；
- generation head 不只是拿 prompt embedding，而是拿经过统一语境建模后的条件；
- 整个系统在最终阶段会做 end-to-end unified SFT。

所以它不是松耦合 pipeline，而更像：

\[
\text{one cognitive core}
\;+\;
\text{one specialized visual renderer}.
\]

### 2. 为什么 MMDiT head 很合理

统一模型常见的一个误区是：一旦追求“统一”，就默认所有输出都该走同一种机制。但文本和图像的生成动力学差异非常大：

- 文本天然适合离散自回归
- 高保真图像更适合连续 latent diffusion / flow 系列方法

因此 InternVL-U 没有为统一而牺牲最合适的视觉生成头，而是接受：

\[
\text{text and image may share cognition,}
\quad
\text{but need not share sampling dynamics}.
\]

这点非常现实，也解释了为什么它在 4B 规模下仍然有不错的性能-效率比。

## 训练配方

### 阶段 1：Generation Head Pre-training

第一阶段冻结 MLLM，只训练 generation head 和相关 projector，在文生图与图像编辑数据上先把视觉合成能力接起来。

这个阶段的目的不是获得完整统一能力，而是：

- 避免一开始就破坏主干已有的理解与推理知识
- 先让生成头学会如何消费来自统一系统的条件信号

### 阶段 2：Any-resolution Continued Pre-training

第二阶段仍然保持 backbone 冻结，但引入 512 到 1024 的可变分辨率训练，增强对不同 aspect ratio 和更复杂版式的适应能力。

这一步很重要，因为很多高语义密度任务本来就依赖：

- 多尺寸文字
- 长宽比变化
- 复杂排版
- 局部区域编辑

如果训练始终停留在单一固定分辨率，模型对 text rendering 与 text editing 的泛化通常会变差。

### 阶段 3：Unified Supervised Finetuning

最后一个阶段解冻整模型，把 CoT 推理数据、图像生成数据、图像编辑数据一起混合，做 end-to-end 优化。

这一步的目标才是真正的 unified capability：

- 主干不只会“看懂”
- generation head 不只会“会画”
- 二者开始学会围绕同一个任务意图协作

## 数据合成与任务设计

### 1. 图像编辑数据的多代理生成

论文为 image editing 构建了多代理（multi-agent）框架，用来生成 instruction-edit pair。按照 HyperAI 对技术报告内容的整理，主要覆盖：

- Global
- Object
- Attribute
- Compositional

这个设计的价值很大，因为它显式增加了编辑操作的结构覆盖面，避免训练集只集中在少数“简单改色”“背景替换”模式。

### 2. 文本渲染数据

论文特别强调 text rendering。其自动化构造思路包括：

- 在自然图像上渲染文本
- 在纯色背景上渲染文本
- 通过自适应 layout 设计构造更真实、更复杂的版式

这说明作者不是把“会生成一张漂亮图”当作统一能力的终点，而是把“能否把抽象文本要求稳定落实为局部视觉细节”当成更难也更关键的问题。

### 3. Text-aware image editing 数据

技术报告中还提到一条三阶段流水线来构造 text-aware image editing 数据：

1. OCR 工具先识别原图中的文字区域与内容
2. MLLM-based instruction agents 生成编辑指令
3. text-editing agents 生成高质量编辑后的配对样本

这条数据管线很有代表性，因为它把“看懂原图文字”“决定要怎么改”“生成改后结果”串成了真正的统一任务。

### 4. Scientific reasoning 与高语义密度生成

论文把 scientific reasoning 也作为重点场景，这透露出一个很重要的研究方向：未来统一模型的难点不再只是 photorealism，而是能否在视觉输出中准确承载概念、规则、因果与布局关系。

## 推理与采样理解

技术报告提到推理阶段采用 Flow-DPM-Solver，并使用 20 步推理。对于条件控制，还使用 classifier-free guidance（CFG）。

若把条件分成完整条件 \(c\) 与无条件分支 \(\varnothing\)，标准 CFG 可以写成

\[
v_{\text{cfg}}
=
v_\varnothing + s \bigl(v_c - v_\varnothing\bigr),
\]

其中 \(s\) 是 guidance scale。

论文还提到对“整个条件丢弃”和“仅文本条件丢弃”分别设置 guidance，这暗示其条件分解更细，至少可以抽象成：

\[
c = (c_{\text{text}}, c_{\text{image}}, c_{\text{other}}).
\]

此时可以做更细粒度的引导，例如比较：

- 完全无条件
- 仅保留图像条件
- 保留全部条件

我对这一点的理解是：InternVL-U 并不把图像编辑视为单一 prompt control，而是在采样阶段继续区分不同条件源的约束强弱。

这里我是在标准 CFG / conditional flow matching 公式上对论文做结构化解释；摘要与公开介绍里确认了 dual-condition guidance 的存在，但没有公开足够多的细节超参数，因此不进一步展开具体数值。

## 直觉 / 理解

我对 InternVL-U 的直觉概括是：

1. 它不是最“纯”的统一模型，而是很“工程上诚实”的统一模型。
2. 它接受文本推理和高保真视觉生成需要不同输出动力学。
3. 它真正想统一的是意图理解、上下文组织和跨任务条件接口。

如果把 Emu3 看成“统一 token 接口”的代表，把 Show-o2 看成“统一主干 + 双头动力学”的代表，那么 InternVL-U 更像：

\[
\text{strong MLLM cognition}
\;+\;
\text{specialized visual generation}
\;+\;
\text{reasoning-centric data synthesis}.
\]

它的贡献不只是一套模型结构，还包括一个现实判断：

\[
\text{better data semantics}
\approx
\text{better unified capability}.
\]

## 与其他方法的关系

### 和 Emu3 的关系

Emu3 追求“next-token prediction is all you need”，统一得更激进；InternVL-U 则认为输出动力学不必完全统一，统一上下文和条件接口就足够重要。

### 和 Janus 的关系

二者都强调理解与生成的视觉冲突，都会做 decoupled visual representations。但 Janus 更像“共享主干、解耦视觉入口”；InternVL-U 则进一步强调如何把强 MLLM 与专门 generation head 协同训练，并用数据合成补上高语义密度能力。

### 和 Show-o2 的关系

Show-o2 也是“共享主干 + 专门视觉生成动力学”的路线，但它更偏 native unified model 的架构设计；InternVL-U 则更明显地建立在一个强现成 MLLM 上，再通过 MMDiT head 与 curriculum training 统一能力。

## 我的笔记 / 开放问题

### 1. 这篇工作的最大价值可能在“数据范式”而不只在结构

很多统一模型论文把重点几乎都放在 backbone / tokenization / objective 上，但 InternVL-U 很清楚地把 text rendering、scientific reasoning、text-aware editing 等高语义密度任务放到中心位置。我很认同这一点，因为很多所谓“生成能力强”的模型其实并不真的擅长执行复杂抽象意图。

### 2. 它的“统一”更像系统级统一，而不是数学形式极简统一

如果从“是不是一个 loss / 一个 token interface”来定义 unified model，那么 InternVL-U 没有 Emu3 那么极端；但如果从“用户是否面对同一个智能体系统”来定义，它反而很像真实可用的一体化方向。

### 3. 仍然值得继续追问的点

- generation head 与 MLLM 主干之间到底共享到什么粒度最优？
- 端到端解冻后，语言 / 推理能力是否会被生成任务拖拽？
- 高语义密度数据的规模、质量和自动评测，是否会成为未来统一模型的主要瓶颈？
- 这类方法继续扩展到 video 时，现有 unified contextual modeling 是否还够用？

## 参考资料

- InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing, arXiv, 2026.
- InternVL-U official repository: https://github.com/OpenGVLab/InternVL-U
- Hugging Face paper page: https://huggingface.co/papers/2603.09877
- HyperAI paper summary page: https://beta.hyper.ai/en/papers/2603.09877
