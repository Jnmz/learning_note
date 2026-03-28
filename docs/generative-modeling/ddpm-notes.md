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

DDPM turns generation into learning the reverse of a Gaussian noising chain, and the familiar epsilon-prediction loss appears because the variational objective collapses to a weighted denoising regression problem.

## Setup

Let \( x_0 \sim q(x_0) \) be a data sample. DDPM defines a forward Markov chain:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t) I \bigr),
\qquad \alpha_t = 1 - \beta_t,
\]

where \( \beta_t \in (0,1) \) is a small variance schedule and \( t = 1, \dots, T \).

Define the cumulative product

\[
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s.
\]

Intuitively:

- each step keeps a fraction \( \sqrt{\alpha_t} \) of the previous signal;
- each step injects fresh Gaussian noise with variance \( 1 - \alpha_t \);
- after many steps, \( x_T \) becomes close to a standard Gaussian.

## Forward Process Closed Form

The first important derivation is that we can sample \( x_t \) from \( x_0 \) in one shot.

Starting from

\[
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0, I),
\]

expand recursively:

\[
\begin{aligned}
x_t
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}\right)
   + \sqrt{1-\alpha_t}\,\epsilon_t \\
&= \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2}
   + \sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1}
   + \sqrt{1-\alpha_t}\,\epsilon_t.
\end{aligned}
\]

Continuing until \( x_0 \), all Gaussian terms remain Gaussian after linear combination, so the result is:

\[
q(x_t \mid x_0) = \mathcal{N}\bigl(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I\bigr).
\]

Equivalently,

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I).
\]

This identity is the algebraic backbone of DDPM training, because it lets us draw a random time \( t \), sample a single \( \epsilon \), and construct the noisy input directly.

## Reverse Model

The generative model tries to reverse the chain:

\[
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t),
\qquad p(x_T)=\mathcal{N}(0,I).
\]

DDPM chooses Gaussian reverse kernels

\[
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}\bigl(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)\bigr).
\]

The central question is: why is learning \( \mu_\theta \) equivalent to predicting noise?

## Posterior Of The Forward Chain

Because the forward process is Gaussian, the posterior

\[
q(x_{t-1}\mid x_t, x_0)
\]

is also Gaussian. Up to constants independent of \( x_{t-1} \),

\[
\log q(x_{t-1}\mid x_t,x_0)
= \log q(x_t\mid x_{t-1}) + \log q(x_{t-1}\mid x_0) + C.
\]

Substitute the two Gaussian densities:

\[
\log q(x_t\mid x_{t-1})
= -\frac{1}{2(1-\alpha_t)}\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2 + C_1,
\]

\[
\log q(x_{t-1}\mid x_0)
= -\frac{1}{2(1-\bar{\alpha}_{t-1})}
\|x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0\|^2 + C_2.
\]

Collect the quadratic and linear terms in \( x_{t-1} \). After completing the square, we get

\[
q(x_{t-1}\mid x_t,x_0)
= \mathcal{N}\bigl(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t I \bigr),
\]

with

\[
\tilde{\mu}_t(x_t,x_0)
= \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+ \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t,
\]

\[
\tilde{\beta}_t
= \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
\]

This matters because the reverse model only needs to approximate this known Gaussian posterior.

## ELBO And The Training Objective

Training maximizes a variational lower bound:

\[
\log p_\theta(x_0)
\ge
\mathbb{E}_q\left[
\log p(x_T)
+ \sum_{t=1}^T \log p_\theta(x_{t-1}\mid x_t)
- \sum_{t=1}^T \log q(x_t\mid x_{t-1})
\right].
\]

Rearranging gives the familiar decomposition

\[
\mathcal{L}_{\text{vlb}}
= \mathbb{E}_q\Bigl[
D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))
+ \sum_{t=2}^T D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))
- \log p_\theta(x_0\mid x_1)
\Bigr].
\]

The key trainable term is the middle KL. If we fix the reverse variance to a known value, each KL becomes a weighted squared error between the true posterior mean and the predicted mean:

\[
\mathcal{L}_t \propto
\mathbb{E}_{q(x_0,\epsilon)}\left[
\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)\|^2
\right].
\]

## Why Epsilon-Prediction Appears

Use the closed form

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
\]

to solve for \( x_0 \):

\[
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}
\left(x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon\right).
\]

Plugging this into \( \tilde{\mu}_t(x_t,x_0) \) and simplifying yields

\[
\tilde{\mu}_t(x_t,x_0)
= \frac{1}{\sqrt{\alpha_t}}
\left(
x_t
- \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon
\right).
\]

So if the network predicts \( \epsilon \) from \( (x_t,t) \), we can construct the reverse mean:

\[
\mu_\theta(x_t,t)
= \frac{1}{\sqrt{\alpha_t}}
\left(
x_t
- \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)
\right).
\]

Substituting this into the KL term shows that, up to a time-dependent weight,

\[
\mathcal{L}_{\text{simple}}
= \mathbb{E}_{t,x_0,\epsilon}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right].
\]

This is the famous DDPM "simple loss". The network does not directly predict pixels of \( x_0 \); it predicts the noise that explains how \( x_0 \) was perturbed into \( x_t \).

## Sampling

Once \( \epsilon_\theta \) is trained, we sample from:

\[
x_{t-1}
= \mu_\theta(x_t,t) + \sigma_t z,
\qquad z \sim \mathcal{N}(0,I),
\]

for \( t=T,T-1,\dots,1 \). This is slow because generation needs many denoising steps, but every step is a relatively stable local prediction.

## Relationship To Score Functions

Since

\[
x_t \mid x_0 \sim \mathcal{N}\bigl(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\bigr),
\]

the conditional score is

\[
\nabla_{x_t}\log q(x_t\mid x_0)
= -\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}
= -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
\]

So predicting noise is equivalent to predicting a scaled score. This is the bridge from DDPM to score-based modeling.

## Practical Reading Summary

- DDPM begins with a discrete Gaussian corruption chain.
- The forward chain has a one-step-from-\(x_0\) closed form.
- The ELBO reduces to matching Gaussian reverse posteriors.
- That mean-matching objective can be reparameterized as epsilon-prediction.
- Epsilon-prediction is secretly score estimation in disguise.

## References

- [Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Zhihu note on diffusion models](https://zhuanlan.zhihu.com/p/689677093)
