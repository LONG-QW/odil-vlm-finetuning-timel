# Finetunning Progress Index

This folder now contains only the assets that have actually entered the SFT / fine-tuning workflow.

## 01_sft_data_build

SFT data construction and conversion stage.

- `finetunning.ipynb`: notebook that writes `odil_timel_sft.jsonl` and prepares train / validation splits.
- `odil_timel_sft.jsonl`: full ID-target SFT JSONL, 9132 rows.
- `build_term_jsonl_from_id_jsonl.py`: converts TIMEL ID targets into term-label targets.
- `split_train_by_test_tsv.py`: split helper.
- `build_test50_from_train.py`: helper used for reduced test-set construction.

## 02_500_sample_training

First real training / smoke-test bundle.

- `odil_500/`: runnable 500-sample ODIL Qwen3-VL SFT bundle.
- `odil_500/odil_dataset_500/train_500.jsonl`: 468 training rows.
- `odil_500/odil_dataset_500/val_500.jsonl`: 32 validation rows.
- `sampled_500_images/`: the 500 sampled images kept beside the bundle.
- `mentor_bundle_odil_500.zip`: packaged 500-sample training bundle.

## 03_full_10000_training

Full dataset training stage.

- `odil_10000/`: formal training dataset with images.
- `odil_10000/train.jsonl`: 8676 ID-target training rows.
- `odil_10000/val.jsonl`: 456 ID-target validation rows.
- `odil_10000/train_terms.jsonl` and `odil_10000/val_terms.jsonl`: term-label variants.
- `train_qwen3vl_sft_timel_merged_fr.py`: production-oriented Qwen3-VL SFT script.
- `train_qwen3vl_sft_timel_merged_fr_V3 2.py`: later V3 training-script iteration.
- `odil_10000.tar`: archived full training dataset.

## 04_random50_eval_split

Evaluation split stage.

- `random50/`: reproducible train / test split package.
- `random50/test.tsv`: fixed-seed random test split.
- `random50/train.tsv`: reduced train split after removing the test set.
- `random50.zip`: packaged split.

## 05_reference_constraints_iteration

Post-training / prediction constraint iteration.

- `id_reference_constraints/`: reference-aware ID and term prediction scripts plus usage notes.

## 99_docs

Fine-tuning notes and slide-revision materials.

- `SFT_ODIL.pdf`
- `ppt_revision/`

## Upstream Stages

The pre-fine-tuning materials are now organized at the project root:

- `01_project_context/`
- `02_source_data/`
- `03_taxonomy_label_mapping/`
- `04_label_distribution_audit/`
- `05_baseline_experiments/`
