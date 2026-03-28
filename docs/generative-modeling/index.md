# Generative Modeling

This section collects notes on models that learn to generate data, with emphasis on the assumptions, objectives, and tradeoffs behind different generative families.

## Scope

- diffusion models
- autoregressive generation
- latent-variable methods
- sampling and guidance

## Notes

- [Diffusion Primer](./diffusion-primer.md): diffusion-family map, terminology, and how the main objectives relate.
- [DDPM Notes](./ddpm-notes.md): the discrete-time forward process, ELBO decomposition, and why epsilon-prediction works.
- [Score Matching Notes](./score-matching-notes.md): what a score is, how Fisher divergence appears, and why diffusion training can be seen as denoising score matching.
- [Flow Matching Notes](./flow-matching-notes.md): probability paths, vector-field regression, and the link to continuous-time generative transport.

## Suggested Future Notes

- latent diffusion
- scaling laws for generative training
