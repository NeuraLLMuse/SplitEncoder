# SplitEncoder

**A single-encoder Transformer architecture for symbolic melodic harmonization and research into generative training methods.**

SplitEncoder is a research implementation of a single-encoder architecture for symbolic music generation. The model processes a melody representation and a partially masked harmony sequence within a shared encoder, enabling harmony generation through iterative unmasking.

The repository provides:

- A reference implementation of the Split Encoder architecture.
- Pretrained checkpoints and reproducible melodic harmonization inference.
- Reference training configurations and evaluation procedures.
- Experimental implementations of alternative masking strategies, diffusion-based training, and training curricula.
- A common foundation for investigating the model's behavior, generalization, and learned representations.

The architecture serves as a foundation for research into symbolic music generation and related applications.