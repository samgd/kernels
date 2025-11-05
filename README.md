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

![RoPE Forward Speed](https://github.com/samgd/kernels/blob/main/assets/rope_fwd.svg?raw=true)

The Triton forward pass kernel launches one program (CUDA block/CTA) per `[batch, seq_len, n_head]` and each program applies RoPE over the `head_dim`. The `[batch, seq_len, n_head, head_dim]` input is loaded and stored from global memory in the native data type (e.g. bfloat16). The cosine and sine arrays used in the rotation have shape `[seq_len, head_dim // 2]` and data type `float32`. Each program loads the `[1, head_dim // 2]` slice at the corresponding `seq_len`. The total cosine and sine loads is therefore `2*[batch, seq_len, n_head, head_dim // 2]` however in practice these arrays fit within cache so global memory loads are limited. 

The `lower` bound on the plot assumes perfect caching so no cosine nor sine global memory loads. The `upper` bound assumes no caching so each program loads the cosine and sine data it needs from global memory.

![RoPE Forward Bandwidth](https://github.com/samgd/kernels/blob/main/assets/rope_fwd_bw.svg?raw=true)

##### Backward

The forward pass of RoPE splits the head_dim into pairs of values and rotates each pair before concatenating them back together. The backwards pass "un-rotates" pairs in the gradient with respect to the output to get the gradient with respect to the input. This "un-rotation" is a fowards pass with the gradient with respect to the output as input and the cached sine rotation array multiplied by -1. 

![RoPE Backward Speed](https://github.com/samgd/kernels/blob/main/assets/rope_bwd.svg?raw=true)