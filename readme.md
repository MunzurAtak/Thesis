# Thesis Debate Pipeline

Code and data for a bachelor thesis comparing three stance-conditioning methods,
**prompting**, **stance-anchored RAG**, and **LoRA fine-tuning**, for preserving
an assigned stance in adversarial multi-round LLM debates.

A *test agent* argues a pro or contra stance against a fixed *opponent* over five
rounds. An LLM *judge* then scores the stance of each turn, and the scores are
used to compute the stance-preservation metrics (Turn of Flip, Number of Flips,
Off-Stance Turn Count). The repository contains the pipeline, the configs for the
final runs, the datasets used, and the final experiment artifacts.

> This repository accompanies the thesis. The grading is based on the thesis
> itself; the code is provided for reference and reproducibility.

## Final experiment

- 3 topics: climate_change, abortion, gun_control
- 2 stances per topic: pro and contra
- 5 seeds: 42, 123, 999, 2024, 2025
- 5 rounds per debate
- 30 debates per condition
- 90 debates total

## Conditions and models

- Prompting: Qwen2.5 7B
- RAG: Qwen2.5 7B + FAISS retrieval
- LoRA: Qwen2.5 7B + trained LoRA adapter
- Opponent: Llama 3.1 8B
- Judge: Gemma 3 12B

The test agent, opponent, and judge are served locally through Ollama.

## Repository structure

```text
configs/                    Configuration files for the final runs
data/                       Datasets used (see DATA.md)
src/                        Pipeline: agents, retrieval, judging, metrics
scripts/                    Experiment runners and analysis/export utilities
final_experiment_artifacts/ Transcripts, judged outputs, metrics, and plots
artifacts/                  Packaged final outputs
```

## Data

All datasets used by the experiment are included in `data/` (raw USDC dataset,
the RAG corpus, the FAISS index, and the LoRA training data). See
[DATA.md](DATA.md) for what each file is, the USDC source and citation, and how
to regenerate the derived data from the raw dataset.

## Running the pipeline

Each command runs one condition end to end (generate debates, judge each turn,
compute metrics):

```bash
python scripts/run_prompting_pipeline.py --config configs/prompting_usdc_3topics_final_eval_final.json
python scripts/run_rag_pipeline.py       --config configs/rag_usdc_faiss_3topics_final_eval_final.json
python scripts/run_lora_pipeline.py      --config configs/lora_usdc_3topics_final_eval_final.json
```

Then export the per-turn scores, drift curves, and summary metrics:

```bash
python scripts/export_analysis_outputs.py --experiment-name prompting_usdc_3topics_final_eval_final
python scripts/export_analysis_outputs.py --experiment-name rag_usdc_faiss_3topics_final_eval_final
python scripts/export_analysis_outputs.py --experiment-name lora_usdc_3topics_final_eval_final
python scripts/recompute_corrected_metrics.py
```

## Requirements

- Python 3.10+ (dependencies in `pyproject.toml`)
- [Ollama](https://ollama.com) with the models above pulled locally
- The LoRA adapter was trained on a Kaggle GPU and merged for local inference
