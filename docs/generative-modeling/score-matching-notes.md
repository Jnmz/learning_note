# Score Matching Notes

## Metadata

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Score matching
  - score-based generative modeling tutorials
  - diffusion tutorials connecting epsilon prediction to scores

## One-Sentence Takeaway

Score matching learns the gradient of log density instead of the density itself, which removes the partition function from the learning target and naturally leads to denoising score matching, diffusion training, and score-based sampling.

## Background / Problem Setup

Suppose our model is an energy-based model

\[
p_\theta(x)=\frac{1}{Z_\theta}\exp(-E_\theta(x)).
\]

Maximum likelihood is often difficult because \( Z_\theta \) is intractable. Score matching avoids this by learning not the density itself, but its gradient with respect to \( x \).

This perspective matters because it is one of the cleanest conceptual bridges from classical energy-based models to modern diffusion and score-based generative modeling.

## Notation

- \( p_{\text{data}}(x) \): data density.
- \( p_\theta(x) \): model density.
- \( E_\theta(x) \): energy function.
- \( Z_\theta \): partition function.
- \( s_\theta(x)=\nabla_x\log p_\theta(x) \): model score.
- \( s_{\text{data}}(x)=\nabla_x\log p_{\text{data}}(x) \): data score.
- \( \nabla\cdot s_\theta(x) \): divergence of the score field.
- \( \sigma \): Gaussian corruption scale.
- \( \tilde{x} \): noisy observation.

## Core Idea

The score of a density \( p(x) \) is

\[
s_p(x)=\nabla_x\log p(x).
\]

It points in the direction of steepest increase of log density. If we know this vector field everywhere, we know a lot about the geometry of the distribution even when the normalized density itself is hard to compute.

The central mathematical story is:

1. differentiating \( \log p_\theta(x) \) removes the partition function;
2. Fisher divergence can be rewritten without the unknown data score;
3. Gaussian corruption yields denoising score matching;
4. diffusion training is a time-indexed version of that denoising objective.

## Detailed Derivation

### Derivation Block 1: Score Definition Removes The Partition Function

Start from

\[
p_\theta(x)=\frac{1}{Z_\theta}\exp(-E_\theta(x)).
\]

Take logs:

\[
\log p_\theta(x)=-E_\theta(x)-\log Z_\theta.
\]

Differentiate with respect to \( x \):

\[
\nabla_x\log p_\theta(x)
= \nabla_x\bigl(-E_\theta(x)-\log Z_\theta\bigr).
\]

Since \( Z_\theta \) depends on parameters but not on the variable \( x \),

\[
\nabla_x \log Z_\theta = 0.
\]

Therefore

\[
s_\theta(x)=\nabla_x\log p_\theta(x)=-\nabla_x E_\theta(x).
\]

This is the first key reason score matching is attractive: it accesses local geometry without requiring the model normalizer.

### Derivation Block 2: Fisher Divergence To Hyvarinen's Objective

The ideal objective is the Fisher divergence

\[
J(\theta)
= \frac{1}{2}\int p_{\text{data}}(x)\|s_\theta(x)-s_{\text{data}}(x)\|^2 dx.
\]

Expand the square:

\[
\begin{aligned}
J(\theta)
&= \frac{1}{2}\int p_{\text{data}}(x)\|s_\theta(x)\|^2 dx \\
&\quad - \int p_{\text{data}}(x)s_\theta(x)^\top s_{\text{data}}(x)\,dx \\
&\quad + \frac{1}{2}\int p_{\text{data}}(x)\|s_{\text{data}}(x)\|^2 dx.
\end{aligned}
\]

The last term does not depend on \( \theta \), so the only nontrivial term is

\[
\int p_{\text{data}}(x)s_\theta(x)^\top s_{\text{data}}(x)\,dx.
\]

Now use the score definition

\[
s_{\text{data}}(x)=\nabla_x\log p_{\text{data}}(x)
= \frac{\nabla_x p_{\text{data}}(x)}{p_{\text{data}}(x)}.
\]

Substitute it:

\[
\int p_{\text{data}}(x)s_\theta(x)^\top s_{\text{data}}(x)\,dx
= \int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\,dx.
\]

Now apply integration by parts coordinatewise. For coordinate \( i \),

\[
\int s_{\theta,i}(x)\,\partial_i p_{\text{data}}(x)\,dx
= \left[s_{\theta,i}(x)p_{\text{data}}(x)\right]_{\partial\Omega}
- \int p_{\text{data}}(x)\,\partial_i s_{\theta,i}(x)\,dx.
\]

Assume the boundary term vanishes. Then

\[
\int s_{\theta,i}(x)\,\partial_i p_{\text{data}}(x)\,dx
= - \int p_{\text{data}}(x)\,\partial_i s_{\theta,i}(x)\,dx.
\]

Summing over coordinates:

\[
\int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\,dx
= -\int p_{\text{data}}(x)\,\nabla\cdot s_\theta(x)\,dx.
\]

Hence the optimization-relevant objective becomes

\[
J(\theta)
= \mathbb{E}_{p_{\text{data}}}
\left[
\frac{1}{2}\|s_\theta(x)\|^2 + \nabla\cdot s_\theta(x)
\right]
+ C.
\]

This is the classical score matching objective. The unknown data score has disappeared.

### Derivation Block 3: Denoising Score Matching With Gaussian Corruption

Plain score matching is hard in high dimension because the divergence term is expensive and the clean-data score may be poorly behaved. So we smooth the data by Gaussian noise:

\[
\tilde{x}=x+\sigma\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

The conditional density is

\[
q_\sigma(\tilde{x}\mid x)
= \mathcal{N}(\tilde{x};x,\sigma^2I).
\]

Take logs:

\[
\log q_\sigma(\tilde{x}\mid x)
= -\frac{d}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}\|\tilde{x}-x\|^2.
\]

Differentiate with respect to \( \tilde{x} \):

\[
\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)
= -\frac{1}{2\sigma^2}\nabla_{\tilde{x}}\|\tilde{x}-x\|^2
= -\frac{\tilde{x}-x}{\sigma^2}.
\]

So the denoising score matching target is known analytically, and the loss becomes

\[
\mathcal{L}_{\mathrm{DSM}}
= \mathbb{E}_{x,\tilde{x}}
\left[
\left\|
s_\theta(\tilde{x},\sigma)
- \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)
\right\|^2
\right]
\]

\[
= \mathbb{E}_{x,\tilde{x}}
\left[
\left\|
s_\theta(\tilde{x},\sigma)
+ \frac{\tilde{x}-x}{\sigma^2}
\right\|^2
\right].
\]

This is the step that makes score estimation practical for modern generative modeling.

## Intuition / Interpretation

- The score is a vector field telling us where density increases.
- Classical score matching tries to match the clean-data vector field directly.
- Denoising score matching first smooths the density, which makes that vector field easier to learn.

I find it helpful to think of denoising score matching as geometry estimation on blurred data manifolds.

## Relation To Other Methods

### Relation To DDPM

Diffusion uses the perturbation

\[
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon.
\]

The corresponding conditional score is

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}.
\]

Since

\[
x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\epsilon,
\]

we obtain

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
\]

So DDPM epsilon-prediction is just score estimation with a deterministic rescaling:

\[
s_\theta(x_t,t)
= -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar{\alpha}_t}}.
\]

This is the direct link to [DDPM Notes](./ddpm-notes.md).

### Relation To Langevin Sampling

Once we have a score estimator, we can sample with Langevin dynamics:

\[
x_{k+1}=x_k+\eta s_\theta(x_k)+\sqrt{2\eta}\,z_k,
\qquad z_k\sim\mathcal{N}(0,I).
\]

The gradient term moves toward high density and the noise term explores the space. This is the classical score-based sampling story.

### Relation To Flow Matching

Flow matching learns a velocity field rather than a score field, but score-based diffusion models also imply a probability-flow ODE. That means a score can induce deterministic transport dynamics. This is the main conceptual bridge to [Flow Matching Notes](./flow-matching-notes.md).

## My Notes / Open Questions

- The most important conceptual jump is not the Fisher divergence itself, but the denoising reformulation.
- Score matching is a very clean "local geometry first" point of view on generative modeling.
- A good future note would compare Langevin, reverse SDE, and probability-flow ODE sampling side by side.

## References

- [Aapo Hyvarinen (2005), *Estimation of Non-Normalized Statistical Models by Score Matching*](https://jmlr.org/papers/v6/hyvarinen05a.html)
- [Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Zhihu note on score-based and diffusion models](https://zhuanlan.zhihu.com/p/12591930520)
