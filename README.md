# Kernels

A library of kernels written in various Python-based languages.

- [Attention](https://github.com/samgd/kernels#attention)
  - [FlashAttention-2](https://github.com/samgd/kernels#flashattention-2)
- [Normalization](https://github.com/samgd/kernels#normalization)
  - [RMSNorm](https://github.com/samgd/kernels#rmsnorm)
- [Position Embedding](https://github.com/samgd/kernels#position-embedding)
  - [Rotary Position Embedding (RoPE)](https://github.com/samgd/kernels#rotary-position-embedding-rope)

## Attention

### FlashAttention-2

Paper: [link](https://arxiv.org/abs/2307.08691)

#### Triton

##### Forward

![Attention Forward Speed](https://github.com/samgd/kernels/blob/main/assets/fa2_fwd.svg?raw=true)

##### Backward

![Attention Backward Speed](https://github.com/samgd/kernels/blob/main/assets/fa2_bwd.svg?raw=true)

## Normalization

### RMSNorm

Paper: [link](https://arxiv.org/abs/1910.07467)

#### Triton

##### Forward

![RMSNorm Forward Speed](https://github.com/samgd/kernels/blob/main/assets/rms_norm_fwd.svg?raw=true)

The Triton forwards pass loads the input vector with shape `[batch_size, n_tokens, hidden_size]` once to compute RMS. It then loads the weight vector of size `[hidden_size,]` and input vector again to compute the normalized output. The normalized output is written back to memory. The total bytes read and written is `element_size * 3*batch_size*n_tokens*hidden_size + hidden_size`. This can be divided by the execution time from the previous plot to compute bandwidth. The 3090 has a maximum bandwidth of ~936GB/s so the achieved bandwidth for smaller hidden sizes must be from hitting cache:

![RMSNorm Forward Bandwidth](https://github.com/samgd/kernels/blob/main/assets/rms_norm_fwd_bw.svg?raw=true)

##### Backward

![RMSNorm Backward Speed](https://github.com/samgd/kernels/blob/main/assets/rms_norm_bwd.svg?raw=true)

## Position Embedding

### Rotary Position Embedding (RoPE)

Paper: [link](https://arxiv.org/abs/2104.09864)

#### Triton

##### Forward

##### Backward