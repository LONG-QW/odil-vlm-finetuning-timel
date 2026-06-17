#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split train JSONL into train/test using an existing TSV file with one column: Image.

Supports paired files (ID and terms) and keeps ordering stable.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_id_jsonl", type=str, required=True)
    p.add_argument("--test_tsv", type=str, required=True)
    p.add_argument("--out_train_id_jsonl", type=str, required=True)
    p.add_argument("--out_test_id_jsonl", type=str, required=True)
    p.add_argument("--train_terms_jsonl", type=str, default="")
    p.add_argument("--out_train_terms_jsonl", type=str, default="")
    p.add_argument("--out_test_terms_jsonl", type=str, default="")
    p.add_argument("--out_train_tsv", type=str, default="")
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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_name_from_example(row: dict) -> str:
    images = row.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError(f"Invalid example without images: {row}")
    return Path(images[0]).name


def read_test_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "Image" not in reader.fieldnames:
            raise ValueError(
                f"TSV must contain an 'Image' column, got: {reader.fieldnames}"
            )
        names = []
        for row in reader:
            name = (row.get("Image") or "").strip()
            if name:
                names.append(name)
    return names


def write_image_tsv(path: str, rows: Sequence[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("Image\n")
        for row in rows:
            f.write(f"{image_name_from_example(row)}\n")


def split_rows(rows: Sequence[dict], test_names: Set[str]) -> tuple[List[dict], List[dict]]:
    train_rows: List[dict] = []
    test_rows: List[dict] = []
    for row in rows:
        name = image_name_from_example(row)
        if name in test_names:
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def main() -> None:
    args = parse_args()

    id_rows = read_jsonl(args.train_id_jsonl)
    if not id_rows:
        raise ValueError(f"Empty train file: {args.train_id_jsonl}")

    requested_names = read_test_names(args.test_tsv)
    requested_set = set(requested_names)
    if len(requested_set) != len(requested_names):
        raise ValueError("Duplicate image names found in test_tsv.")

    present_names = {image_name_from_example(row) for row in id_rows}
    missing = sorted(requested_set - present_names)
    if missing:
        raise ValueError(
            f"{len(missing)} images from test_tsv are not present in train JSONL. "
            f"Examples: {missing[:10]}"
        )

    train_id_rows, test_id_rows = split_rows(id_rows, requested_set)
    write_jsonl(args.out_train_id_jsonl, train_id_rows)
    write_jsonl(args.out_test_id_jsonl, test_id_rows)

    print(
        f"[ID] total={len(id_rows)} train={len(train_id_rows)} test={len(test_id_rows)} "
        f"from_tsv={args.test_tsv}"
    )

    if args.out_train_tsv:
        write_image_tsv(args.out_train_tsv, train_id_rows)
        print(f"[TSV] wrote train split list: {args.out_train_tsv}")

    if args.train_terms_jsonl:
        if not args.out_train_terms_jsonl or not args.out_test_terms_jsonl:
            raise ValueError(
                "When --train_terms_jsonl is set, out terms paths are required."
            )
        term_rows = read_jsonl(args.train_terms_jsonl)
        if len(term_rows) != len(id_rows):
            raise ValueError(
                f"ID/terms row count mismatch: {len(id_rows)} vs {len(term_rows)}"
            )
        train_term_rows, test_term_rows = split_rows(term_rows, requested_set)
        write_jsonl(args.out_train_terms_jsonl, train_term_rows)
        write_jsonl(args.out_test_terms_jsonl, test_term_rows)
        print(
            f"[TERMS] total={len(term_rows)} train={len(train_term_rows)} "
            f"test={len(test_term_rows)} from_tsv={args.test_tsv}"
        )


if __name__ == "__main__":
    main()
