# random50

This folder is the reproducible split package for the current dataset.

Files:
- `build_test50_from_train.py`: script used to sample 50 test images with a fixed seed.
- `train.tsv`: image-name list for the reduced train split.
- `test.tsv`: image-name list for the 50 sampled test split.

Sampling settings:
- `n_test = 50`
- `seed = 42`

Notes:
- This folder keeps only the minimal files needed for the reproducible random split request.
- If needed later, the script can also regenerate the corresponding JSONL files from the source dataset.
