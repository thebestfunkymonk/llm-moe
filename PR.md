## PR: Bug fixes, robustness improvements, and post-training sanity check

### Summary
- Fix tokenizer/dataset loading (remove invalid args, avoid pickling tokenizer)
- Make cached data robust across environments (store tokenizer name, not object)
- Add a post-training generation sanity check with a simple sampling loop
- Guard Muon orthogonalization to avoid bfloat16 matmul on CPU
- Set AMP usage based on CUDA availability to prevent CPU autocast issues
- Update documentation and requirements

### Details

1) Data loading and caching
- Removed invalid `token=False` arguments from `AutoTokenizer.from_pretrained` and `load_dataset` calls that could raise runtime errors.
- Replaced pickled tokenizer object with a `tokenizer_name` string in cache. On load, we reconstruct the tokenizer from its name and set `pad_token` if missing. This avoids pickling issues and improves portability.

2) AMP usage safety
- `train_model` and `train_moe_model` relied on `autocast()` contexts when `config.use_amp` is True. On CPU, this can be problematic. We now set `use_amp` in the top-level models to `torch.cuda.is_available()` so AMP is only enabled on CUDA.
- `evaluate_model` already used `autocast(enabled=...)`; left intact.

3) Muon optimizer stability on CPU
- `zeropower_via_newtonschulz5` previously forced `bfloat16`. On some CPUs, `bfloat16` matmul can be unsupported or slow. We now use `bfloat16` only on CUDA and fall back to `float32` on CPU to avoid errors.

4) Post-training sanity check
- Implemented `generate_text(model, tokenizer, prompt, ...)` and call it after finishing training each model with a short prompt. This provides a quick end-to-end functional check that logits map to plausible tokens.

5) Docs and dependencies
- README: document the new generation sanity check.
- requirements.txt: add missing `torch`, `numpy`, and `tqdm` which are imported by the code but were not listed.

### Notes on MoE Implementation
- The router returns top-k indices and weights; tokens are dispatched per-expert using boolean masking. Weighted outputs are accumulated per token; load-balancing auxiliary loss encourages more uniform usage. The approach is simple and workable for small-scale experiments.

### Future Improvements (optional)
- Add expert capacity and token dropping to avoid pathological overload.
- Use a better load-balancing loss (e.g., auxiliary from Switch/GShard) for stability.
- Support checkpoint save/load and a minimal CLI interface for generation.
