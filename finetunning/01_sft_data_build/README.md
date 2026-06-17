# SFT Data Build

This stage contains the notebook and helper scripts that produce SFT JSONL files.

Run `finetunning.ipynb` from the repository root (`/Users/longquanwen0813/haka`) because the notebook reads the upstream source files from:

- `02_source_data/data/all.tsv`
- `02_source_data/data/classes.tsv`

The main generated artifact kept here is:

- `odil_timel_sft.jsonl` with 9132 rows

The later formal train / validation split is kept in:

- `../03_full_10000_training/odil_10000/`
