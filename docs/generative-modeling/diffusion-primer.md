# Diffusion Primer

## Metadata

- Topic: generative-modeling
- Status: evergreen
- Last updated: 2026-03-29
- Source type: concept
- Primary references:
  - Denoising Diffusion Probabilistic Models
  - Score-based tutorials and score matching references
  - Flow Matching for Generative Modeling

## One-Sentence Takeaway

Diffusion-style models all rely on a simple idea: choose an easy-to-analyze path from data to noise, then learn the inverse dynamics that bring noise back to the data distribution.

## Why It Matters

This family matters because it turns hard density modeling into local prediction problems: predict noise, predict a score, or predict a velocity field. Those local objectives are often stable, scalable, and compatible with conditioning.

## Core Ideas

### One Generative Family, Three Common Parameterizations

- **DDPM view**: define a discrete Markov noising chain and learn the reverse chain.
- **Score-based view**: learn the score \( \nabla_x \log p_t(x) \) of progressively noised data and integrate a reverse SDE or ODE.
- **Flow-matching view**: define a probability path \( p_t \) and regress the vector field whose ODE transports noise to data.

These are not isolated ideas. In many practical setups:

- epsilon-prediction in DDPM can be converted to a score estimate;
- the probability-flow ODE of score-based models gives a deterministic transport view;
- flow matching directly trains such a transport field without simulating the forward diffusion chain during optimization.

### Common Structure

Most notes in this topic fit the same template:

1. pick a path \( p_t \) from data to a simple prior;
2. derive a local target attached to each noisy state \( x_t \);
3. integrate a reverse-time process or ODE at sampling time.

## Important Details

### A Minimal Dictionary

- **Forward process**: the chosen data-to-noise path.
- **Reverse process**: the learned dynamics that move samples from noise back to data.
- **Score**: \( \nabla_x \log p_t(x) \), the steepest ascent direction of log density.
- **Velocity field**: \( v_t(x) \), the instantaneous motion in an ODE \( \dot{x}_t = v_t(x_t) \).
- **Probability path**: the family of intermediate marginals \( \{p_t\}_{t \in [0,1]} \).

### Which Note To Read Next

- If you want the original discrete-time derivation, start with [DDPM Notes](./ddpm-notes.md).
- If you want to understand why diffusion objectives are tied to scores, read [Score Matching Notes](./score-matching-notes.md).
- If you want the modern ODE transport perspective, read [Flow Matching Notes](./flow-matching-notes.md).

### Visual Map

![Diffusion overview](./diffusion-process.svg)

## Personal Notes

I find it useful to treat diffusion, score-based models, and flow matching as three coordinate systems on the same terrain. The algebra looks different, but the central question is the same: what local signal lets a neural network reconstruct global generation dynamics?

## Open Questions

- Which probability paths make vector-field learning easiest?
- When is a stochastic reverse process actually better than a deterministic ODE solver?

## See Also

- [Unified Models](../unified-models/index.md)

## References

- [Ho, Jain, Abbeel (2020), *Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239)
- [Lilian Weng, *What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution*](https://yang-song.net/blog/2021/score/)
- [Lipman et al. (2023), *Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747)
