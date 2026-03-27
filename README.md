# DISCO: Resolving the Identity Crisis in Text-to-Image Generation

## Abstract

State-of-the-art text-to-image models suffer from a persistent identity crisis when generating scenes with multiple humans, frequently producing duplicate faces, merged identities, or incorrect person counts. This limitation undermines realism and reduces the usefulness of such models in applications requiring accurate multi-person synthesis.

We present **DISCO (Reinforcement with Diversity Constraints)**, a reinforcement learning framework that directly optimizes identity diversity both within a single image and across groups of generated samples. DISCO fine-tunes flow-matching text-to-image models using **Group-Relative Policy Optimization (GRPO)** and a compositional reward that penalizes intra-image facial similarity, discourages cross-sample identity repetition, enforces accurate person counts, and preserves visual quality and prompt alignment through human preference scores. Training is stabilized using a single-stage curriculum that gradually increases prompt complexity.

DISCO requires no real training data and establishes cross-sample diversity as a critical axis for resolving identity collapse. On the DiverseHumans benchmark, DISCO achieves **98.6% Unique Face Accuracy** and near-perfect **Global Identity Spread**, outperforming both open-source and proprietary systems while maintaining high perceptual quality. These results position DISCO as a scalable, annotation-free solution for robust multi-human image synthesis.
