# Reference Constraints

This folder is reserved for referential / controlled-vocabulary work around TIMEL outputs.

Current scripts:
- ID version: /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/scripts/train_qwen3vl_sft_timel_with_reference.py
- term version: /Users/longquanwen0813/haka/finetunning/05_reference_constraints_iteration/id_reference_constraints/scripts/train_qwen3vl_sft_timel_with_term_reference.py

Important:
- the original reference-aware script is the ID version
- the new term version mirrors the same idea for labels / keywords
- both versions currently use post-generation filtering / normalization
- neither version is yet token-level constrained decoding during generation

Subfolders:
- scripts: code for referential loading, constrained prediction, normalization
- data: whitelist/reference files or derived artifacts used only for this task
- predictions: generated predictions and cleaned outputs
- docs: notes and short run instructions
