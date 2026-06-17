# ID Referential Constraint Usage

Main script:
- /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/scripts/train_qwen3vl_sft_timel_with_reference.py

Current scope:
- keeps the V3 training pipeline intact
- adds `classes.tsv` loading as a TIMEL referential
- adds `predict` mode
- cleans generated IDs against the whitelist from `classes.tsv`
- drops invalid IDs and removes duplicates while preserving order

Important:
- this is a practical whitelist-based constraint after generation
- it is not yet full token-level constrained decoding

Reference file:
- /Users/longquanwen0813/haka/02_source_data/data/classes.tsv

Example: prediction with referential cleaning

```bash
python3 /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/scripts/train_qwen3vl_sft_timel_with_reference.py \
  --mode predict \
  --model_name /path/to/checkpoint/or/final \
  --predict_jsonl /path/to/test.jsonl \
  --classes_tsv /Users/longquanwen0813/haka/02_source_data/data/classes.tsv \
  --pred_out /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/predictions/preds.jsonl \
  --precision bf16 \
  --device_map auto
```

Output JSONL fields:
- `image`
- `raw_prediction`
- `predicted_ids`
- `predicted_labels`
- `invalid_ids`
- `gold_ids` (if the input JSONL already contains assistant labels)
