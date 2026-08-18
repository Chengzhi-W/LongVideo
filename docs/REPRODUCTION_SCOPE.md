# LongVideo Memory-DMD reproduction scope

This branch preserves the experiments that were actually run, rather than
mixing them with abandoned 4-GPU, 5-second, rank-offset-seed or step-1200
variants.

Core comparison configurations:

- senior baseline: `use_memory_dmd: false`
- senior mem1 non-normalized: `A18 - A0`, scale 1, all timesteps
- Long-vs-Local no gate: `A18 - A6`, joint DMD normalization, all timesteps
- Long-vs-Local t<750: `A18 - A6`, joint DMD normalization, memory applied only
  to shifted teacher timesteps below 750

The current inference model is unchanged by `A18-A6`; the two AR-teacher
branches only construct a training target. The local branch is freshly
prefilled from six recent latents at absolute RoPE positions 12..17; it is not
sliced from the full-history KV cache. Both branches predict the identical
noisy current chunk at positions 18..20.

The UltraVideo-Long scripts prepare either 21- or 63-latent clean data. The
long AR-teacher configuration explicitly expands Wan's sequence capacity to 63
latents and trains through the 63-latent teacher-forcing attention mask. A
one-GPU, low-spatial-resolution smoke test must pass before the two-node run;
only the production-resolution fine-tune should be used for DMD experiments.
