# VLM Reading Map

## Metadata

- Topic: vlm
- Status: seed
- Last updated: 2026-03-29
- Source type: concept
- Primary references:
  - Foundational VLM pretraining and instruction-tuning papers

## One-Sentence Takeaway

Vision-language models can be organized usefully by how they align vision and text, where they fuse information, and what supervision they rely on.

## Why It Matters

The VLM literature has grown quickly, and simple categories make it easier to compare design choices without losing the big picture.

## Core Ideas

### Alignment Strategy

Some systems use contrastive alignment, while others train generatively or rely on instruction-following supervision.

### Fusion Location

Models differ in whether they fuse early, late, or through a lightweight connector into a pretrained language model.

### Capability Framing

It is useful to separate perception-heavy tasks from reasoning-heavy tasks when reading new VLM papers.

## Important Details

- Architecture: vision encoder plus text encoder or LLM backbone
- Objective: contrastive, captioning, next-token prediction, or instruction tuning
- Data: image-text pairs, QA data, OCR-heavy corpora, synthetic instruction data
- Evaluation: captioning, VQA, grounding, document tasks, agent-style reasoning
- Strengths: rich supervision and practical multimodal interfaces
- Limitations: brittle grounding, hallucination, and benchmark sensitivity

## Personal Notes

This page works well as a hub for later notes on CLIP-like, Flamingo-like, and LLM-connected VLM families.

## Open Questions

- Which benchmarks still measure real multimodal reasoning rather than language priors?
- How much of recent progress comes from data curation versus architecture?

## See Also

- [Unified Multimodal Models Overview](../unified-models/unified-multimodal-models-overview.md)

## References

- Add concrete paper notes here as you build out the section.

