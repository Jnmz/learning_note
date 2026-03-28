# DDPM Notes

## Metadata

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Denoising Diffusion Probabilistic Models
  - Lilian Weng's diffusion overview
  - user-supplied Zhihu notes

## One-Sentence Takeaway

DDPM defines a discrete Gaussian noising chain and learns its reverse, and the familiar epsilon-prediction loss is a reparameterized form of a variational objective built from Gaussian posterior matching.

## Background / Problem Setup

We want to model samples \( x_0 \sim q(x_0) \). DDPM does not try to learn \( q(x_0) \) in one step. Instead, it introduces a latent chain

\[
x_0 \to x_1 \to \cdots \to x_T
\]

where the forward process gradually corrupts data into noise and the reverse process learns to undo this corruption.

The overall strategy is:

1. choose a simple forward Gaussian Markov chain;
2. write a reverse model with learnable Gaussian transitions;
3. optimize a variational lower bound;
4. reparameterize the trainable part as a noise-prediction problem.

## Notation

- \( x_0 \): clean data sample.
- \( x_t \): latent variable at diffusion step \( t \in \{1,\dots,T\} \).
- \( q(x_t\mid x_{t-1}) \): forward noising kernel.
- \( p_\theta(x_{t-1}\mid x_t) \): learned reverse kernel.
- \( \beta_t \in (0,1) \): variance schedule.
- \( \alpha_t = 1-\beta_t \).
- \( \bar{\alpha}_t = \prod_{s=1}^t \alpha_s \).
- \( \epsilon,\epsilon_t \sim \mathcal{N}(0,I) \): Gaussian noise variables.
- \( \mu_\theta(x_t,t) \): reverse mean.
- \( \Sigma_\theta(x_t,t) \): reverse covariance.

## Core Idea

The forward chain is chosen as

\[
q(x_t\mid x_{t-1})
= \mathcal{N}\bigl(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\bigr).
\]

Each step keeps a scaled version of the previous state and injects Gaussian noise. After many steps, \( x_T \) becomes close to standard Gaussian. The reverse model then uses

\[
p_\theta(x_{t-1}\mid x_t)
= \mathcal{N}\bigl(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\bigr)
\]

to reconstruct data from noise.

The central derivation is:

- first derive the direct forward marginal \( q(x_t\mid x_0) \);
- then derive the forward posterior \( q(x_{t-1}\mid x_t,x_0) \);
- finally show why the ELBO reduces to epsilon prediction.

## Detailed Derivation

### Derivation Block 1: Closed Form Of \( q(x_t \mid x_0) \)

Start from the forward reparameterization

\[
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal{N}(0,I).
\]

Expand one step:

\[
\begin{aligned}
x_t
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-1}\right)
+ \sqrt{1-\alpha_t}\epsilon_t \\
&= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
+ \sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}
+ \sqrt{1-\alpha_t}\epsilon_t.
\end{aligned}
\]

Continuing recursively gives

\[
x_t
= \sqrt{\bar{\alpha}_t}x_0
+ \sum_{s=1}^t
\left(
\sqrt{1-\alpha_s}\prod_{r=s+1}^t \sqrt{\alpha_r}
\right)\epsilon_s.
\]

This is a linear combination of independent Gaussian variables, so conditional on \( x_0 \), it is Gaussian. Its mean is

\[
\mathbb{E}[x_t\mid x_0]=\sqrt{\bar{\alpha}_t}x_0.
\]

Its covariance is

\[
\operatorname{Var}(x_t\mid x_0)
= \sum_{s=1}^t
\left(
(1-\alpha_s)\prod_{r=s+1}^t \alpha_r
\right)I.
\]

Now use the telescoping identity

\[
\sum_{s=1}^t
\left(
(1-\alpha_s)\prod_{r=s+1}^t \alpha_r
\right)
= 1-\prod_{s=1}^t \alpha_s
= 1-\bar{\alpha}_t.
\]

Therefore

\[
q(x_t\mid x_0)
= \mathcal{N}\bigl(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\bigr).
\]

Equivalently, by Gaussian reparameterization,

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

This formula is crucial because it lets us sample \( x_t \) from \( x_0 \) in one shot.

### Derivation Block 2: Posterior \( q(x_{t-1}\mid x_t,x_0) \)

We now derive the exact reverse conditional of the forward chain. By Bayes' rule and the Markov property,

\[
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0).
\]

The two Gaussian factors are

\[
q(x_t\mid x_{t-1})
\propto
\exp\left(
-\frac{1}{2(1-\alpha_t)}\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2
\right),
\]

\[
q(x_{t-1}\mid x_0)
\propto
\exp\left(
-\frac{1}{2(1-\bar{\alpha}_{t-1})}
\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
\right).
\]

Expand the quadratic terms in \( x_{t-1} \):

\[
\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2
= \|x_t\|^2 - 2\sqrt{\alpha_t}x_t^\top x_{t-1} + \alpha_t\|x_{t-1}\|^2,
\]

\[
\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2
= \|x_{t-1}\|^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0^\top x_{t-1}
+ \bar{\alpha}_{t-1}\|x_0\|^2.
\]

Keeping only terms that depend on \( x_{t-1} \), the log density is

\[
\log q(x_{t-1}\mid x_t,x_0)
=
-\frac{1}{2}
\left[
\frac{\alpha_t}{1-\alpha_t}
+ \frac{1}{1-\bar{\alpha}_{t-1}}
\right]\|x_{t-1}\|^2
\]

\[
\qquad
+ \left[
\frac{\sqrt{\alpha_t}}{1-\alpha_t}x_t
+ \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
\right]^\top x_{t-1}
+ C.
\]

Now simplify the quadratic coefficient using \( \bar{\alpha}_t=\alpha_t\bar{\alpha}_{t-1} \) and \( 1-\alpha_t=\beta_t \):

\[
\frac{\alpha_t}{1-\alpha_t}
+ \frac{1}{1-\bar{\alpha}_{t-1}}
= \frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}.
\]

Completing the square gives

\[
q(x_{t-1}\mid x_t,x_0)
= \mathcal{N}\bigl(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t I\bigr),
\]

where

\[
\tilde{\beta}_t
= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
\]

\[
\tilde{\mu}_t(x_t,x_0)
= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t.
\]

So the reverse model only needs to approximate a known Gaussian posterior mean and variance.

### Derivation Block 3: From ELBO To Epsilon Prediction

The reverse generative model is

\[
p_\theta(x_{0:T})
= p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),
\qquad p(x_T)=\mathcal{N}(0,I).
\]

Using \( q(x_{1:T}\mid x_0) \) as a variational distribution and Jensen's inequality,

\[
\log p_\theta(x_0)
= \log \int q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}dx_{1:T}
\ge
\mathbb{E}_q\left[
\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
\]

Expanding numerator and denominator yields

\[
\mathcal{L}_{\mathrm{vlb}}
= \mathbb{E}_q\left[
\log p(x_T)
+ \sum_{t=1}^T \log p_\theta(x_{t-1}\mid x_t)
- \sum_{t=1}^T \log q(x_t\mid x_{t-1})
\right].
\]

Rearranging terms by conditional distributions gives

\[
\mathcal{L}_{\mathrm{vlb}}
= \mathbb{E}_q\Bigl[
D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))
+ \sum_{t=2}^T D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\bigr)
- \log p_\theta(x_0\mid x_1)
\Bigr].
\]

If the reverse variance is fixed, each KL term reduces to a weighted mean-squared error:

\[
\mathcal{L}_t
\propto
\mathbb{E}\left[
\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)\|^2
\right].
\]

Now use the direct forward marginal

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
\]

to solve for \( x_0 \):

\[
x_0
= \frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
\right).
\]

Substitute this into \( \tilde{\mu}_t(x_t,x_0) \):

\[
\begin{aligned}
\tilde{\mu}_t(x_t,x_0)
&=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}
\cdot
\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
\right)
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \\
&=
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}x_t
- \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}\epsilon
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t.
\end{aligned}
\]

Combine the \( x_t \) coefficients:

\[
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
= \frac{1}{\sqrt{\alpha_t}}.
\]

Therefore

\[
\tilde{\mu}_t(x_t,x_0)
= \frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon
\right).
\]

If the network predicts \( \epsilon_\theta(x_t,t) \), define

\[
\mu_\theta(x_t,t)
= \frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right).
\]

Then the trainable loss becomes, up to a time-dependent weight,

\[
\mathcal{L}_{\mathrm{simple}}
= \mathbb{E}_{t,x_0,\epsilon}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right].
\]

This is why DDPM is usually implemented as a noise-prediction model even though it started from an ELBO.

## Intuition / Interpretation

- The forward chain manufactures many supervised denoising tasks from unlabeled data.
- The reverse posterior says what the ideal denoiser should do if it had access to \( x_0 \).
- Epsilon-prediction works because \( x_0 \) and \( \epsilon \) are interchangeable once \( x_t \) and \( t \) are known.

I find DDPM easiest to understand as "a Gaussian latent-variable model with a very carefully chosen auxiliary chain."

## Relation To Other Methods

### Relation To Score Matching

Since

\[
x_t\mid x_0 \sim \mathcal{N}\bigl(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\bigr),
\]

the score of the conditional kernel is

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}.
\]

Using \( x_t-\sqrt{\bar{\alpha}_t}x_0=\sqrt{1-\bar{\alpha}_t}\epsilon \), we get

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
\]

So predicting \( \epsilon \) is equivalent to predicting a scaled score. This is the precise bridge to [Score Matching Notes](./score-matching-notes.md).

### Relation To Flow Matching

DDPM is a discrete stochastic model; flow matching is a continuous transport model. But both fit the same high-level pattern:

- choose a path from data to noise;
- define a local supervision signal along the path;
- integrate learned reverse dynamics at sampling time.

In DDPM the local signal is noise or posterior mean. In flow matching it is velocity. See [Flow Matching Notes](./flow-matching-notes.md).

## My Notes / Open Questions

- The cleanest conceptual move in DDPM is not the network design but the Gaussian algebra behind the forward posterior.
- The ELBO story explains the original method well, but many practical improvements later prioritize better parameterizations over the exact bound.
- A natural follow-up note would compare DDPM, DDIM, and probability-flow ODE sampling in one place.

## References

- [Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Zhihu note on diffusion models](https://zhuanlan.zhihu.com/p/689677093)
