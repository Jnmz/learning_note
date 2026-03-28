# Flow Matching Notes

## Metadata

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: derivation note
- Primary references:
  - Flow Matching for Generative Modeling
  - diffusion and score-based tutorials for comparison

## One-Sentence Takeaway

Flow matching learns a time-dependent velocity field whose ODE transports a simple base distribution to the data distribution, and its training objective works because conditional velocity regression recovers the marginal transport field by conditional expectation.

## Background / Problem Setup

DDPM and score-based models often start from stochastic noising processes. Flow matching starts from the transport viewpoint instead:

1. choose a probability path \( \{p_t\}_{t\in[0,1]} \) between a simple source distribution and the data distribution;
2. describe that path by a velocity field;
3. learn the velocity field directly by regression.

This is attractive because it makes generation look like continuous-time mass transport rather than reverse-time denoising.

## Notation

- \( p_0 \): source distribution, often \( \mathcal{N}(0,I) \).
- \( p_1 \): target data distribution.
- \( p_t \): intermediate marginal density at time \( t\in[0,1] \).
- \( x_t \): state of a particle at time \( t \).
- \( v_t(x) \): marginal velocity field.
- \( p_t(x\mid x_0,x_1) \): conditional probability path.
- \( u_t(x\mid x_0,x_1) \): conditional velocity field.
- \( \mu_t(x_0,x_1) \): conditional path mean.
- \( \sigma_t \): conditional path scale.
- \( \pi(dx_0,dx_1) \): endpoint coupling.

## Core Idea

Flow matching starts from the ODE

\[
\frac{dx_t}{dt}=v_t(x_t).
\]

If particles follow this ODE and \( x_0\sim p_0 \), then the induced density evolves from \( p_0 \) to \( p_1 \). The hard part is learning \( v_t \). Flow matching solves this by constructing a conditional path where the target velocity is analytically tractable.

## Detailed Derivation

### Derivation Block 1: From ODE Dynamics To The Continuity Equation

Assume particles follow

\[
\dot{x}_t=v_t(x_t).
\]

Let \( p_t(x) \) denote the density of \( x_t \). For any smooth test function \( \varphi(x) \),

\[
\frac{d}{dt}\mathbb{E}_{p_t}[\varphi(x)]
= \frac{d}{dt}\int \varphi(x)p_t(x)\,dx.
\]

By the chain rule along trajectories,

\[
\frac{d}{dt}\varphi(x_t)
= \nabla \varphi(x_t)^\top \dot{x}_t
= \nabla \varphi(x_t)^\top v_t(x_t).
\]

Taking expectation gives

\[
\frac{d}{dt}\int \varphi(x)p_t(x)\,dx
= \int \nabla \varphi(x)^\top v_t(x)p_t(x)\,dx.
\]

Now apply integration by parts:

\[
\int \nabla \varphi(x)^\top v_t(x)p_t(x)\,dx
= -\int \varphi(x)\,\nabla\cdot\bigl(p_t(x)v_t(x)\bigr)\,dx,
\]

assuming the boundary term vanishes.

Therefore

\[
\int \varphi(x)\,\partial_t p_t(x)\,dx
= -\int \varphi(x)\,\nabla\cdot\bigl(p_t(x)v_t(x)\bigr)\,dx.
\]

Since this holds for all test functions \( \varphi \), we obtain the continuity equation:

\[
\partial_t p_t(x)+\nabla\cdot\bigl(p_t(x)v_t(x)\bigr)=0.
\]

This equation is the density-level conservation law behind flow matching.

### Derivation Block 2: Conditional Gaussian Path And Conditional Velocity

Choose a conditional path between endpoints \( x_0 \sim p_0 \) and \( x_1 \sim p_1 \):

\[
p_t(x\mid x_0,x_1)
= \mathcal{N}\bigl(x;\mu_t(x_0,x_1),\sigma_t^2 I\bigr).
\]

Use the reparameterization

\[
x_t = \mu_t(x_0,x_1) + \sigma_t \epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

Differentiate with respect to time:

\[
\frac{dx_t}{dt}
= \partial_t \mu_t(x_0,x_1)+\dot{\sigma}_t\epsilon.
\]

Now solve the reparameterization for \( \epsilon \):

\[
\epsilon = \frac{x_t-\mu_t(x_0,x_1)}{\sigma_t}.
\]

Substitute back:

\[
u_t(x_t\mid x_0,x_1)
= \partial_t \mu_t(x_0,x_1)
+ \frac{\dot{\sigma}_t}{\sigma_t}\bigl(x_t-\mu_t(x_0,x_1)\bigr).
\]

This is the conditional velocity target. It is available in closed form because the conditional path was chosen to be analytically simple.

For the common linear mean path

\[
\mu_t(x_0,x_1)=(1-t)x_0 + tx_1,
\]

we have

\[
\partial_t \mu_t(x_0,x_1)=x_1-x_0,
\]

so

\[
u_t(x_t\mid x_0,x_1)
= (x_1-x_0)
+ \frac{\dot{\sigma}_t}{\sigma_t}\bigl(x_t-\mu_t(x_0,x_1)\bigr).
\]

This derivation explicitly uses reparameterization and variable substitution.

### Derivation Block 3: Why Conditional Velocity Regression Recovers The Marginal Field

The training objective is

\[
\mathcal{L}_{\mathrm{FM}}
= \mathbb{E}\left[
\|v_\theta(x_t,t)-u_t(x_t\mid x_0,x_1)\|^2
\right],
\]

where \( x_t\sim p_t(\cdot\mid x_0,x_1) \).

Fix time \( t \) and location \( x \). The pointwise regression problem is

\[
\min_v \mathbb{E}\left[
\|v-u_t(x_t\mid x_0,x_1)\|^2 \mid x_t=x
\right].
\]

Expand the square:

\[
\mathbb{E}\left[
\|v-u\|^2 \mid x_t=x
\right]
= \|v\|^2 - 2v^\top \mathbb{E}[u\mid x_t=x] + \mathbb{E}[\|u\|^2\mid x_t=x].
\]

Differentiate with respect to \( v \) and set to zero:

\[
2v - 2\mathbb{E}[u\mid x_t=x] = 0.
\]

Hence the optimal predictor is

\[
v_t^\star(x)
= \mathbb{E}\bigl[u_t(x_t\mid x_0,x_1)\mid x_t=x\bigr].
\]

Now check that this field transports the marginal path. The marginal density is

\[
p_t(x)=\int p_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1).
\]

Differentiate with respect to time and use the conditional continuity equation:

\[
\partial_t p_t(x)
= \int \partial_t p_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1)
\]

\[
= -\int \nabla\cdot\bigl(p_t(x\mid x_0,x_1)u_t(x\mid x_0,x_1)\bigr)\,\pi(dx_0,dx_1).
\]

Move divergence outside the integral:

\[
\partial_t p_t(x)
= -\nabla\cdot\left(
\int p_t(x\mid x_0,x_1)u_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1)
\right).
\]

By the definition of conditional expectation,

\[
\int p_t(x\mid x_0,x_1)u_t(x\mid x_0,x_1)\,\pi(dx_0,dx_1)
= p_t(x)\,v_t^\star(x).
\]

Therefore

\[
\partial_t p_t(x)+\nabla\cdot\bigl(p_t(x)v_t^\star(x)\bigr)=0.
\]

So the regression optimum is exactly the marginal velocity field that realizes the full probability path.

## Intuition / Interpretation

- The continuity equation says probability mass is conserved as particles move.
- The conditional path gives us easy local supervision signals.
- Conditional expectation stitches those local signals into the correct marginal transport field.

I find this much easier to reason about once I stop thinking in terms of denoising and start thinking in terms of moving probability mass.

## Relation To Other Methods

### Relation To DDPM

DDPM uses a discrete stochastic chain and learns reverse denoising transitions. Flow matching uses a continuous-time probability path and learns a velocity field. Both are path-based generative methods, but the local supervision differs:

- DDPM target: noise or posterior mean.
- flow-matching target: velocity.

See [DDPM Notes](./ddpm-notes.md) for the discrete Gaussian-chain perspective.

### Relation To Score Matching

Score-based models learn

\[
s_t(x)=\nabla_x\log p_t(x).
\]

For an SDE

\[
dx = f(x,t)\,dt + g(t)\,dW_t,
\]

the associated probability-flow ODE is

\[
\frac{dx}{dt}
= f(x,t) - \frac{1}{2}g(t)^2\nabla_x\log p_t(x).
\]

So a score field induces a deterministic velocity field. Flow matching starts directly from that ODE-style viewpoint rather than deriving it from an SDE first. This is the main bridge to [Score Matching Notes](./score-matching-notes.md).

### ODE, SDE, And Continuity Equation

- ODE: describes deterministic particle motion.
- SDE: describes stochastic particle motion with diffusion noise.
- continuity equation: describes how the density evolves under the chosen dynamics.

These are complementary descriptions at the trajectory level, stochastic level, and density level.

## My Notes / Open Questions

- The conditional expectation proof is the conceptual center of flow matching; without it, the method can look like an arbitrary regression trick.
- Flow matching feels especially clarifying when comparing deterministic transport to score-based probability-flow ODEs.
- A useful future note would compare rectified flow, optimal transport flow matching, and classical score-based probability-flow ODEs.

## References

- [Lipman et al. (2023), *Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/)
