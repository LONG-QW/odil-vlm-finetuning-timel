# ODIL 500-sample Training Bundle

## Contents
- `train_qwen3vl_sft_timel_merged_fr.py`
- `odil_dataset_500/train_500.jsonl`
- `odil_dataset_500/val_500.jsonl`
- `odil_dataset_500/data/images/*` (500 images)
- `requirements_odil_qwen.txt`

## Environment
- Python 3.10.x (tested: 3.10.19)
- GPU recommended for full training
- Install dependencies:

```bash
python3 -m pip install -r requirements_odil_qwen.txt
```

## Remote Quick Check (recommended first)
Run a 1-step smoke test before full training:

```bash
HF_DATASETS_CACHE=./.hf_datasets_cache \
python3 train_qwen3vl_sft_timel_merged_fr.py \
  --train_jsonl odil_dataset_500/train_500.jsonl \
  --val_jsonl odil_dataset_500/val_500.jsonl \
  --output_dir qwen3_vl_500_smoke \
  --max_steps 1 \
  --per_device_bs 1 \
  --grad_accum 1 \
  --precision bf16 \
  --device_map auto \
  --resume false
```

## Full Run
```bash
HF_DATASETS_CACHE=./.hf_datasets_cache \
python3 train_qwen3vl_sft_timel_merged_fr.py \
  --train_jsonl odil_dataset_500/train_500.jsonl \
  --val_jsonl odil_dataset_500/val_500.jsonl \
  --output_dir qwen3_vl_500_out \
  --precision bf16 \
  --device_map auto \
  --resume false
```

## Notes
- JSONL image paths are `data/images/...` and resolved from JSONL directory.
- This subset contains 468 train + 32 val samples.
- If the server cannot access Hugging Face, download model weights beforehand and pass a local model path via `--model_name`.
- Ensure output disk space is sufficient (checkpoints can be large).
