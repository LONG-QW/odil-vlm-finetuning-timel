#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a random held-out test set by removing N examples from train JSONL.

Supports paired files (ID and terms) to keep the same sampled indices.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_id_jsonl", type=str, required=True)
    p.add_argument("--out_train_id_jsonl", type=str, required=True)
    p.add_argument("--out_test_id_jsonl", type=str, required=True)
    p.add_argument("--train_terms_jsonl", type=str, default="")
    p.add_argument("--out_train_terms_jsonl", type=str, default="")
    p.add_argument("--out_test_terms_jsonl", type=str, default="")
    p.add_argument("--out_train_tsv", type=str, default="")
    p.add_argument("--out_test_tsv", type=str, default="")
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def image_name_from_example(row: dict) -> str:
    images = row.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError(f"Invalid example without images: {row}")
    return Path(images[0]).name


def write_image_tsv(path: str, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Image\n")
        for row in rows:
            f.write(f"{image_name_from_example(row)}\n")


def split_by_indices(rows: Sequence[dict], test_idx: set[int]) -> tuple[List[dict], List[dict]]:
    train_rows: List[dict] = []
    test_rows: List[dict] = []
    for i, r in enumerate(rows):
        if i in test_idx:
            test_rows.append(r)
        else:
            train_rows.append(r)
    return train_rows, test_rows


def main() -> None:
    args = parse_args()

    id_rows = read_jsonl(args.train_id_jsonl)
    n = len(id_rows)
    if n == 0:
        raise ValueError(f"Empty train file: {args.train_id_jsonl}")
    if not (1 <= args.n_test < n):
        raise ValueError(f"n_test must be in [1, {n-1}], got {args.n_test}")

    rng = random.Random(args.seed)
    all_idx = list(range(n))
    test_idx = set(rng.sample(all_idx, args.n_test))

    id_train, id_test = split_by_indices(id_rows, test_idx)
    write_jsonl(args.out_train_id_jsonl, id_train)
    write_jsonl(args.out_test_id_jsonl, id_test)

    print(f"[ID] total={n} train={len(id_train)} test={len(id_test)} seed={args.seed}")

    if args.out_train_tsv:
        write_image_tsv(args.out_train_tsv, id_train)
        print(f"[TSV] wrote train split list: {args.out_train_tsv}")
    if args.out_test_tsv:
        write_image_tsv(args.out_test_tsv, id_test)
        print(f"[TSV] wrote test split list: {args.out_test_tsv}")

    if args.train_terms_jsonl:
        if not args.out_train_terms_jsonl or not args.out_test_terms_jsonl:
            raise ValueError(
                "When --train_terms_jsonl is set, out terms paths are required."
            )
        term_rows = read_jsonl(args.train_terms_jsonl)
        if len(term_rows) != n:
            raise ValueError(
                f"ID/terms row count mismatch: {n} vs {len(term_rows)}"
            )
        term_train, term_test = split_by_indices(term_rows, test_idx)
        write_jsonl(args.out_train_terms_jsonl, term_train)
        write_jsonl(args.out_test_terms_jsonl, term_test)
        print(
            f"[TERMS] total={n} train={len(term_train)} test={len(term_test)} seed={args.seed}"
        )


if __name__ == "__main__":
    main()
