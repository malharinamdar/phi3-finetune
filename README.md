# phi3-finetune

Fine-tuning Microsoft's Phi-3-mini-4k-instruct on medical question answering, using parameter-efficient methods and preference optimization.

## Overview

This project fine-tunes Phi-3-mini on multiple-choice medical reasoning questions (USMLE-style) in two stages:

1. **Supervised fine-tuning (SFT)** — QLoRA adapters trained on prompt/response pairs, with the base model loaded in 4-bit (bitsandbytes, nf4).
2. **Preference optimization** — a GRPO/DPO-style stage on `preference_pairs.json`, a set of `prompt` / `chosen` / `rejected` triples, to push the model toward better-reasoned answers.

The pipeline uses `transformers`, `peft` (LoRA), `trl`, and `bitsandbytes` for 4-bit quantized training, with evaluation on held-out medical QA prompts.

## Files

- `phi3-finetune.ipynb` — the full pipeline: setup, 4-bit model loading, LoRA configuration, SFT, preference optimization, and evaluation.
- `preference_pairs.json` — JSONL of preference pairs (`prompt`, `chosen`, `rejected`) used for the preference-optimization stage.

## Setup

```bash
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece
```

A CUDA GPU is required for 4-bit training. The notebook is written to run on a single GPU (e.g. Colab / cloud instance).

## Note

This is an experimental fine-tuning project. The notebook is exploratory and includes the iterations involved in getting QLoRA + preference training working with Phi-3.
