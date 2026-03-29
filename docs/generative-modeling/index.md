# 生成建模

这一部分收集学习“如何生成数据”的模型笔记，重点关注不同生成模型家族背后的假设、目标函数与方法权衡。

## 范围

- 扩散模型
- 自回归生成
- 潜变量方法
- 采样与引导

## 笔记

- [扩散模型导读](./diffusion-primer.md)：梳理扩散家族的基本地图、术语，以及主要目标函数之间的联系。
- [DDPM 笔记](./ddpm-notes.md)：系统推导前向高斯加噪、后验均值、ELBO 分解与 epsilon-prediction。
- [Score Matching 笔记](./score-matching-notes.md)：系统推导 Fisher divergence、Hyvarinen 目标、denoising score matching 以及它与扩散训练的联系。
- [Flow Matching 笔记](./flow-matching-notes.md)：系统推导 continuity equation、条件高斯路径、条件速度回归与 probability-flow ODE。
- [VAE 与 ELBO 笔记](./vae-elbo-notes.md)：系统推导变分推断、ELBO 分解、高斯重参数化与闭式 KL。

## 后续可补充的笔记

- latent diffusion
- 生成训练中的 scaling laws
