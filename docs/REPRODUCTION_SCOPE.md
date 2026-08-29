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

The current UltraVideo-Long AR-teacher path prepares 81-latent clean data from
321 pixel frames. The long AR-teacher configuration expands Wan's sequence
capacity to 81 latents and trains with the locked
`sink3 + slow12 + fast3 + noisy-current3` attention contract. A one-GPU,
low-spatial-resolution smoke test must pass before production. Production may
use one node with 8 GPUs and accumulation 2, or two nodes with 8 GPUs each and
accumulation 1; both keep effective global batch 16. Only the
production-resolution fine-tune should be used for downstream DMD experiments.
