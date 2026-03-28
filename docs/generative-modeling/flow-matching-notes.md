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

Flow matching learns a time-dependent vector field whose ODE transports a simple base distribution into the data distribution, so training becomes direct regression on target velocities along a chosen probability path.

## ODE View Of Generation

Instead of a reverse Markov chain, flow matching starts from an ODE:

\[
\frac{d x_t}{dt} = v_t(x_t),
\qquad t \in [0,1].
\]

If we sample \( x_0 \sim p_0 \) from a simple base distribution and integrate the ODE to \( t=1 \), the pushforward distribution becomes \( p_1 \), which we want to match to the data distribution.

The unknown object is the vector field \( v_t(\cdot) \).

## Continuity Equation

The density induced by the ODE evolves according to the continuity equation:

\[
\partial_t p_t(x) + \nabla \cdot \bigl(p_t(x) v_t(x)\bigr) = 0.
\]

This equation is the density-level analogue of moving particles with velocity \( v_t \).

Read it as conservation of mass:

- \( \partial_t p_t(x) \) measures local density change;
- \( \nabla \cdot (p_t v_t) \) measures how probability mass flows in or out.

## Probability Paths

Flow matching assumes we choose an interpolation path \( p_t \) between:

- \( p_0 \): a tractable source distribution such as \( \mathcal{N}(0,I) \);
- \( p_1 \): the data distribution.

The training problem is then:

"Find a vector field whose induced dynamics realize this path."

This separates model design into two pieces:

1. choose a path \( p_t \);
2. learn the vector field that is consistent with it.

## Conditional Probability Paths

The paper's practical trick is to define a path conditioned on a data sample \( x_1 \). Let \( x_0 \sim p_0 \) and \( x_1 \sim q \) be paired through some coupling. For each pair, define a conditional path \( p_t(x \mid x_0, x_1) \).

Then define a conditional velocity field \( u_t(x \mid x_0, x_1) \) that exactly generates that path.

The model \( v_\theta(x,t) \) is trained by regression:

\[
\mathcal{L}_{\text{FM}}
= \mathbb{E}_{t, x_0, x_1, x_t}
\left[
\|v_\theta(x_t,t)-u_t(x_t\mid x_0,x_1)\|^2
\right],
\]

where \( x_t \sim p_t(\cdot \mid x_0,x_1) \).

This looks simple, but there is a conceptual leap hidden inside: the target is a velocity, not a score or a noise term.

## Why Conditional Regression Solves The Marginal Problem

The model only sees \( x_t \) and \( t \), not the endpoints \( x_0, x_1 \). Why does regressing toward conditional velocities help?

The key identity is conditional expectation. The optimal mean-squared predictor is

\[
v_t^\star(x)
= \mathbb{E}[u_t(x_t \mid x_0,x_1)\mid x_t=x].
\]

So the trained field is the average conditional velocity at location \( x \) and time \( t \). Under the construction in the flow-matching paper, this averaged field is precisely the marginal field that transports the full path \( p_t \).

This is why the objective is practical:

- the conditional target \( u_t \) is analytically available;
- the marginal target would be hard to write directly;
- regression recovers the needed marginal field automatically.

## The Gaussian Path Example

One common choice is a Gaussian bridge-style path:

\[
p_t(x \mid x_0,x_1)
= \mathcal{N}\bigl(x; \mu_t(x_0,x_1), \sigma_t^2 I\bigr),
\]

with a mean path such as

\[
\mu_t(x_0,x_1) = (1-t)x_0 + t x_1.
\]

If we reparameterize

\[
x_t = \mu_t(x_0,x_1) + \sigma_t \epsilon,
\qquad \epsilon \sim \mathcal{N}(0,I),
\]

then differentiating with respect to time gives the conditional velocity target:

\[
u_t(x_t \mid x_0,x_1)
= \partial_t \mu_t(x_0,x_1) + \frac{\dot{\sigma}_t}{\sigma_t}\bigl(x_t-\mu_t(x_0,x_1)\bigr).
\]

This formula is worth pausing on:

- \( \partial_t \mu_t \) moves the mean along the interpolation path;
- the second term rescales deviations around the mean as the Gaussian width changes.

When \( \sigma_t \) shrinks toward zero near the data endpoint, the flow becomes increasingly focused.

## Relation To Diffusion And Score Models

There are several close connections.

### 1. Both Learn Local Fields

- diffusion often learns noise \( \epsilon_\theta \);
- score models learn \( s_\theta = \nabla_x \log p_t(x) \);
- flow matching learns velocity \( v_\theta \).

All three are local geometric signals attached to \( x_t \).

### 2. Probability-Flow ODE Bridge

Score-based diffusion models have an associated probability-flow ODE. That means a score model can also be interpreted as defining a deterministic transport field.

Flow matching starts directly from this transport viewpoint instead of deriving it from a stochastic process.

### 3. Training Differences

Diffusion training is tied to a noising process and often to specific variance schedules.

Flow matching is more path-centric:

- choose a probability path;
- derive its target velocity;
- regress the field directly.

This can simplify training and make the connection to optimal transport more explicit.

## Practical Summary

- choose a source distribution \( p_0 \);
- choose an interpolation path \( p_t \);
- derive the conditional velocity target \( u_t \);
- fit \( v_\theta(x,t) \) with squared loss;
- sample by solving the learned ODE from \( t=0 \) to \( t=1 \).

## When This Perspective Helps

I find flow matching especially clarifying when diffusion notation starts to feel overloaded. It strips the story down to transport:

- what path do particles follow?
- what instantaneous velocity moves them?
- how do local motions accumulate into a global generative map?

That makes it a useful bridge between diffusion models, continuous normalizing flows, and optimal transport-flavored formulations.

## References

- [Lipman et al. (2023), *Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/)
