# FAIRMT Reproduction with Steering Vectors

This repository is based on the FAIRMT benchmark for multi-turn fairness evaluation in LLMs.

It reproduces parts of the original pipeline and extends it with activation steering vector experiments using Llama-based models.

## Modifications

- Added activation steering vectors to the generation pipeline
- Replaced original model with Llama-3.1-8B-Instruct
- Used Llama Guard 3 for evaluation
- Added scripts for running multiple steering-vector experiments

## Running the Code

Generate responses:
```bash
python all_answers.py
```

Set steering vectors in:
```bash
generate_answers_vectors.py
```

Evaluate results:
```bash
python evaluate_all.py
```

## Files

- `outputs_*` → generated model outputs
- `evals_*` → evaluation results
- `Llama-3.1-8B-instruct` → steering vectors folder

## Findings

Results are detailed in:
- `PR_findings.pdf`

## Reference

Original FAIRMT benchmark:
https://arxiv.org/abs/2410.19317
