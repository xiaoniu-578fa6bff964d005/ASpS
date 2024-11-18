# Accelerated Speculative Sampling Based on Tree Monte Carlo

An implementation of Accelerated Speculative Sampling (ASpS) for faster LLM inference.

## Usage

```python
import lib

# Choose one of three sampling methods:
# 1. Basic sampling (no speculation)
# 2. Speculative Sampling (SpS)
# 3. Accelerated Speculative Sampling (ASpS)
generator = lib.tmc_sample_generator(
    model=target_model,
    ref_model=reference_model,
    input_ids=input_ids,
    n=num_tokens, 
    process_logits_kwargs={"logits_processor": logits_processor}
)

# Generator yields (tokens, logprobs) at each step
for tokens, logprobs in generator:
    # Process generated tokens
    ...
```

## Key Functions

- `basic_sample_generator`: Standard autoregressive sampling
- `mc_sample_generator`: Original Speculative Sampling (SpS) implementation
- `tmc_sample_generator`: Our proposed Accelerated Speculative Sampling (ASpS)

## Citation

```bibtex
@inproceedings{huaccelerated,
  title={Accelerated Speculative Sampling Based on Tree Monte Carlo},
  author={Hu, Zhengmian and Huang, Heng},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```
