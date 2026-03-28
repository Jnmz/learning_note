# Generative Modeling

This section collects notes on models that learn to generate data, with emphasis on the assumptions, objectives, and tradeoffs behind different generative families.

## Scope

- diffusion models
- autoregressive generation
- latent-variable methods
- sampling and guidance

## Notes

- [Diffusion Primer](./diffusion-primer.md): diffusion-family map, terminology, and how the main objectives relate.
- [DDPM Notes](./ddpm-notes.md): a structured derivation note covering forward Gaussian noising, posterior mean algebra, ELBO decomposition, and epsilon-prediction.
- [Score Matching Notes](./score-matching-notes.md): a structured derivation note covering Fisher divergence, Hyvarinen's objective, denoising score matching, and the bridge to diffusion training.
- [Flow Matching Notes](./flow-matching-notes.md): a structured derivation note covering the continuity equation, conditional Gaussian paths, conditional velocity regression, and probability-flow ODE connections.

## Suggested Future Notes

- latent diffusion
- scaling laws for generative training
