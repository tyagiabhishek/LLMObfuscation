# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproducing "RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?" (arXiv:2506.14261) by Gupta & Jenner. Original code: https://github.com/cybershiptrooper/RL_Obfuscation

## Notebook Structure

`part1_linear_probes.ipynb` — 4-section reproduction notebook:
1. **Dataset Preparation**: BeaverTails safe/harmful pairs + RL prompt set
2. **Initial Probe Training**: Extract activations → train per-layer linear probes → evaluate TPR@1%FPR
3. **RL-Obfuscation**: GRPO via TRL with LoRA, combined reward (probe evasion + quality + length)
4. **Post-Obfuscation Evaluation**: Original probes on obfuscated model + retrained probes

## Key Technical Details

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (paper uses Llama 2 7B / Llama 3 8B / Qwen 14B)
- Dataset: `PKU-Alignment/BeaverTails` (paper uses Mechanistic-Anomaly-Detection datasets)
- Activations: PyTorch forward hooks on `model.model.layers`, response tokens only
- Probes: `nn.Linear(d_model, 1)` with BCE, 40 epochs, batch 16 (matches paper)
- RL: GRPO with β=0.05, lr=1e-5, 8 generations/prompt, LoRA r=16
- Reward: λ_wb=0.8 (threshold), λ_bb=1.0 (keyword quality proxy), λ_length=2.0
- Eval metric: TPR at 1% FPR

## Simplifications vs Paper

- Smaller model (1.5B vs 7-14B)
- Keyword-based blackbox reward instead of StrongReject Gemma-2B classifier
- 200 prompts instead of 1000
- No curriculum training (progressive threshold tightening)

## Dependencies

```
transformers datasets accelerate torch scikit-learn matplotlib numpy tqdm trl peft
```

## Paper Reference Values

- Probe layers: L10 (Llama2), L12 (Llama3), L20 (Qwen14B)
- Baseline TPR@1%FPR: ~96% (linear, mean aggregation)
- Post-obfuscation evasion: ~60% (mean), ~0% (max-pool)
- GRPO config: temperature=1.0, KL β=0.05, lr=1e-5, 8 completions/prompt
