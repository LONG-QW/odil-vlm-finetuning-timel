# Term Referential Constraint Usage

Main script:
- /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/scripts/train_qwen3vl_sft_timel_with_term_reference.py

Current scope:
- keeps the V3 training pipeline intact
- uses `classes.tsv` labels as a controlled vocabulary
- adds `predict` mode
- normalizes generated terms after generation
- matches labels with case/accent-insensitive normalization
- removes duplicates and drops out-of-vocabulary terms

Important:
- this is a whitelist-based term constraint after generation
- it is not yet full constrained decoding during generation

Reference file:
- /Users/longquanwen0813/haka/02_source_data/data/classes.tsv

Example: prediction with term referential cleaning

```bash
python3 /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/scripts/train_qwen3vl_sft_timel_with_term_reference.py \
  --mode predict \
  --model_name /path/to/checkpoint/or/final \
  --predict_jsonl /path/to/test_terms.jsonl \
  --classes_tsv /Users/longquanwen0813/haka/02_source_data/data/classes.tsv \
  --pred_out /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/predictions/term_preds.jsonl \
  --precision bf16 \
  --device_map auto
```

Output JSONL fields:
- `image`
- `raw_prediction`
- `predicted_terms`
- `predicted_ids`
- `invalid_terms`
- `gold_terms`
- `gold_ids`
