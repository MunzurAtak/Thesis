# Thesis Debate Pipeline

This repository contains the final code for a bachelor thesis experiment comparing Prompting, stance-anchored RAG, and LoRA for stance preservation in adversarial multi-round LLM debates.

## Final experiment

- 3 topics: climate_change, abortion, gun_control
- 2 stances per topic: pro and contra
- 3 seeds: 42, 123, 999
- 5 rounds per debate
- 18 debates per condition
- 54 debates total

## Conditions

- Prompting: Qwen2.5 7B
- RAG: Qwen2.5 7B + FAISS retrieval
- LoRA: Qwen2.5 7B + trained LoRA adapter
- Adversary: Llama 3.1 8B
- Judge: Gemma 3 12B

## Repository structure

```text
configs/     Final experiment configuration files
src/         Core pipeline, agents, retrieval, judging, and metrics code
scripts/     Experiment runners and analysis/export utilities
artifacts/   Final packaged experiment outputs
```

## Running the final experiments

```powershell
python scripts/run_prompting_pipeline.py --config configs/prompting_usdc_3topics_final_eval_final.json
python scripts/run_rag_pipeline.py --config configs/rag_usdc_faiss_3topics_final_eval_final.json
python scripts/run_lora_pipeline.py --config configs/lora_usdc_3topics_final_eval_final.json
```

## Exporting results

```powershell
python scripts/export_analysis_outputs.py --experiment-name prompting_usdc_3topics_final_eval_final
python scripts/export_analysis_outputs.py --experiment-name rag_usdc_faiss_3topics_final_eval_final
python scripts/export_analysis_outputs.py --experiment-name lora_usdc_3topics_final_eval_final
```

## Final artifacts

Final outputs are stored in:

```text
artifacts/final_experiment_artifacts.zip
```

This archive contains final configs, metrics, plots, transcripts, judged transcripts, judge validation files, and the RAG leakage audit.
