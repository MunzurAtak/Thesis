# Data

This thesis uses a single external dataset, **USDC** (User Stance and Dogmatism
in Conversations), plus data derived from it for the RAG and LoRA conditions.
All of these files are included in this repository so the experiment can be
inspected and reproduced.

## Source dataset: USDC

> Mounika Marreddy, Subba Reddy Oota, Venkata Charan Chinni, Manish Gupta, and
> Lucie Flek. *USDC: A Dataset of User Stance and Dogmatism in Long
> Conversations.* Findings of the Association for Computational Linguistics:
> ACL 2025, pp. 23715–23759.

- Paper: https://aclanthology.org/2025.findings-acl.1216/
- Official dataset and code: https://github.com/mounikamarreddy/USDC

USDC contains 764 multi-user Reddit conversation threads and 9,618 stance-labelled
posts on a five-point stance scale. It is used here only as a source of
topically relevant, stance-labelled text. The dataset is redistributed in this
repository for thesis reproducibility; for licensing and terms please refer to
the official USDC repository above.

## What is in `data/`

### `data/raw/` — raw USDC dataset
- `USDC_Stance.pkl` — stance split (pandas pickle).
- `USDC_Dogmatism.pkl` — dogmatism split (pandas pickle).
- `USDC_Stance_csv.zip` — zipped copy of `USDC_Stance.csv` (the CSV form read by
  the corpus builder). The uncompressed CSV is ~174 MB, above GitHub's per-file
  limit, so it is committed zipped. **Unzip it before rebuilding the corpus:**
  ```bash
  cd data/raw && unzip USDC_Stance_csv.zip
  ```

### `data/rag_corpus/` — RAG corpus built from USDC
- `usdc_rag_corpus.json` — full corpus built from USDC.
- `usdc_selected_rag_corpus.json` — the selected subset that is actually indexed
  and retrieved during the RAG condition.

### `data/rag_indexes/` — retrieval index used at inference
- `usdc_selected.faiss` — FAISS index over the selected corpus.
- `usdc_selected_metadata.json` — passage metadata aligned to the index.

### `data/lora_training/` — synthetic LoRA training data
Stance-consistent chat-style examples anchored to the three topics, used to train
the LoRA adapter. `usdc_3topics_controlled_lora_{train,val}.jsonl` are the final
controlled files used in the thesis.

## Regenerating the derived data from raw

```bash
# 1. Unzip the raw CSV
cd data/raw && unzip USDC_Stance_csv.zip && cd ../..

# 2. Build the RAG corpus from USDC
python scripts/build_selected_usdc_rag_corpus.py

# 3. Build the FAISS index over the selected corpus
python scripts/build_faiss_index.py

# 4. Build the controlled LoRA training data
python scripts/build_controlled_lora_training_data.py
```

## Note on stance labels

The RAG corpus maps each USDC post's stance label onto the assigned debate
stance. USDC annotates a user's stance toward the Reddit submission under
discussion rather than toward the debate proposition used in this thesis, so the
retrieved passages are best read as topically related, approximately
stance-labelled material. This is discussed as a limitation in the thesis.
