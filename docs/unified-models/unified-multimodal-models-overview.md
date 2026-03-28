# Unified Multimodal Models Overview

## Metadata

- Topic: unified-models
- Status: seed
- Last updated: 2026-03-29
- Source type: concept
- Primary references:
  - Survey papers and model families spanning joint multimodal representation and generation

## One-Sentence Takeaway

Unified multimodal models try to reduce task-specific fragmentation by representing diverse inputs and outputs inside a more shared modeling framework.

## Why It Matters

A unified setup can simplify transfer across tasks, reduce duplicated modeling components, and make multimodal reasoning look more like a general sequence modeling problem.

## Core Ideas

### Shared Representation

Many unified systems push different modalities into compatible token or latent spaces so one backbone can process them together.

### Shared Objective

A common loss family, often autoregressive prediction or masked reconstruction, can create a single training interface across tasks.

### Shared Interface

Prompting, instruction tuning, and general input-output formatting often matter as much as architecture when the goal is broad capability.

## Important Details

- Architecture: often a transformer backbone with modality-specific encoders or decoders
- Objective: autoregressive, denoising, contrastive, or mixed multitask losses
- Data: mixed text, image, video, and instruction datasets
- Evaluation: transfer breadth, multimodal reasoning, and generation quality
- Strengths: reuse, transfer, and simpler system framing
- Limitations: optimization interference, token imbalance, and evaluation ambiguity

## Personal Notes

This is a good umbrella topic page to split into narrower notes once the repository grows.

## Open Questions

- When does unification improve generalization versus simply increasing optimization difficulty?
- Which interface choices matter more than backbone choices?

## See Also

- [VLM Reading Map](../vlm/vlm-reading-map.md)

## References

- Add targeted model or survey references here as this section expands.

