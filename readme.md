# Thesis Debate Pipeline

This repository contains the code for my bachelor thesis experiment on stance preservation in multi-round LLM debates.

The thesis compares three stance-conditioning methods:

1. Prompting
2. Stance-anchored Retrieval-Augmented Generation (RAG)
3. LoRA fine-tuning

The current implemented pipeline focuses on the prompting baseline.

---

## Current status

Implemented:

- Prompting test agent
- Fixed adversarial opponent agent
- Debate environment
- Config-based experiment running
- Transcript saving as JSON
- Transcript validation
- Ollama LLM backend
- Ollama judge
- Strict and polarity stance metrics
- Turn-of-Flip (ToF)
- Number-of-Flips (NoF)
- Turn-level judge score export
- Drift-curve CSV export
- Drift-curve plot generation

Not yet implemented:

- RAG agent
- LoRA agent
- Full topic selection from USDC
- Final statistical analysis

---

## Setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
python -m pip install -e .
```

Install Ollama and pull the local model:

```powershell
ollama pull qwen2.5:3b
```

---

## Fast smoke test

Use this to quickly check whether the full pipeline works:

```powershell
python scripts/run_prompting_pipeline.py --config configs/prompting_ollama_judge_smoke_test.json
```

This runs:

- 1 topic
- 1 stance pair
- 1 debate round
- Ollama generation
- Ollama judge scoring
- metrics export

---

## Full prompting debug run

Use this to run the current 5-round prompting baseline debug experiment:

```powershell
python scripts/run_prompting_pipeline.py --config configs/prompting_ollama_judge_debug.json
```

This creates:

```text
outputs/transcripts/
outputs/judge_scores/
outputs/metrics/
```

---

## Export analysis outputs

After running the pipeline, export turn scores, drift-curve data, and drift-curve plot:

```powershell
python scripts/export_analysis_outputs.py --experiment-name prompting_ollama_judge_debug
```

This creates:

```text
outputs/metrics/prompting_ollama_judge_debug_turn_scores.csv
outputs/metrics/prompting_ollama_judge_debug_drift_curve.csv
outputs/plots/prompting_ollama_judge_debug_drift_curve.png
```

---

## Re-score existing transcripts

If only the judge prompt changed, do not regenerate debates. Reuse existing transcripts:

```powershell
python scripts/score_existing_transcripts.py --config configs/prompting_ollama_judge_debug.json
python scripts/export_analysis_outputs.py --experiment-name prompting_ollama_judge_debug
```

This saves runtime because debate generation is skipped.

---

## Judge calibration test

To test whether the judge handles basic stance cases correctly:

```powershell
python scripts/test_judge_calibration.py
```

Expected behavior:

- Pro-ban utterances should receive +1 or +2.
- Anti-ban utterances should receive -1 or -2.
- Descriptive/neutral utterances should receive 0.

---

## Output structure

```text
outputs/
├── transcripts/       raw debate transcripts
├── judge_scores/      judged transcripts
├── metrics/           ToF, NoF, turn scores, drift data
└── plots/             generated figures
```

---

## Main metrics

### Strict stance consistency

Any weakening of the assigned stance counts as a flip.

Example:

```text
assigned = +2
judged = +1
strict flip = yes
```

### Polarity stance consistency

Only neutralization or movement to the opposite side counts as a flip.

Example:

```text
assigned = +2
judged = +1
polarity flip = no
```

### Turn-of-Flip

The first debate round where stance inconsistency occurs.

### Number-of-Flips

The total number of inconsistent turns in a debate.

### Drift curve

Mean judged stance score over debate rounds.

---

## Current workflow

For normal development:

```powershell
python scripts/run_prompting_pipeline.py --config configs/prompting_ollama_judge_smoke_test.json
```

For full prompting baseline debugging:

```powershell
python scripts/run_prompting_pipeline.py --config configs/prompting_ollama_judge_debug.json
python scripts/export_analysis_outputs.py --experiment-name prompting_ollama_judge_debug
```

For judge-only changes:

```powershell
python scripts/score_existing_transcripts.py --config configs/prompting_ollama_judge_debug.json
python scripts/export_analysis_outputs.py --experiment-name prompting_ollama_judge_debug
```
