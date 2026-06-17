#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert ODIL SFT JSONL targets from TIMEL IDs to TIMEL term labels.

Input JSONL format (one example):
{
  "images": ["data/images/xxxx.jpg"],
  "messages": [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": [{"type":"text","text":"tm-a, tm-b"}]}
  ]
}

Output keeps the same schema and image paths, but assistant text becomes labels:
"Christ, ange, Vierge, ..."
"""

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument(
        "--classes_tsv",
        type=str,
        default="/Users/longquanwen0813/haka/02_source_data/data/classes.tsv",
        help="TSV with columns: timel_id, timel_label",
    )
    p.add_argument(
        "--prompt_text",
        type=str,
        default="Générer les mots-clés timel pour cette image.",
    )
    p.add_argument(
        "--unknown_policy",
        type=str,
        choices=["keep", "drop", "error"],
        default="keep",
        help="How to handle unknown timel_id in classes.tsv.",
    )
    return p.parse_args()


def load_id2label(classes_tsv: str) -> Dict[str, str]:
    id2label: Dict[str, str] = {}
    with open(classes_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"classes.tsv has no header: {classes_tsv}")
        for col in ("timel_id", "timel_label"):
            if col not in reader.fieldnames:
                raise ValueError(
                    f"classes.tsv missing required column '{col}', got: {reader.fieldnames}"
                )
        for row in reader:
            tid = (row.get("timel_id") or "").strip()
            label = (row.get("timel_label") or "").strip()
            if tid and label:
                id2label[tid] = label
    if not id2label:
        raise ValueError(f"No valid id/label pairs loaded from {classes_tsv}")
    return id2label


def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return " ".join(parts)
    return ""


def set_text_in_content(content, new_text: str):
    if isinstance(content, str):
        return new_text
    if isinstance(content, list):
        new_content = []
        replaced = False
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text" and not replaced:
                q = dict(p)
                q["text"] = new_text
                new_content.append(q)
                replaced = True
            else:
                new_content.append(p)
        if not replaced:
            new_content.append({"type": "text", "text": new_text})
        return new_content
    return [{"type": "text", "text": new_text}]


def split_ids(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def convert_example(
    ex: dict,
    id2label: Dict[str, str],
    prompt_text: str,
    unknown_policy: str,
) -> Tuple[dict, List[str], int, int]:
    out = copy.deepcopy(ex)
    messages = out.get("messages", [])

    # Update user prompt text for term-generation task
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            for i, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    q = dict(part)
                    q["text"] = prompt_text
                    content[i] = q
                    break

    unknown_ids: List[str] = []
    total_ids = 0
    mapped_ids = 0

    # Convert assistant id list -> label list
    for m in messages:
        if m.get("role") != "assistant":
            continue

        old_text = extract_text_from_content(m.get("content"))
        ids = split_ids(old_text)
        total_ids += len(ids)

        labels: List[str] = []
        for tid in ids:
            label = id2label.get(tid)
            if label is not None:
                labels.append(label)
                mapped_ids += 1
                continue

            unknown_ids.append(tid)
            if unknown_policy == "keep":
                labels.append(tid)
            elif unknown_policy == "drop":
                pass
            else:
                raise KeyError(f"Unknown timel_id in assistant text: {tid}")

        new_text = ", ".join(labels)
        m["content"] = set_text_in_content(m.get("content"), new_text)

    out["messages"] = messages
    return out, unknown_ids, total_ids, mapped_ids


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    id2label = load_id2label(args.classes_tsv)

    total_examples = 0
    total_ids = 0
    mapped_ids = 0
    unknown_pool: List[str] = []

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            new_ex, unknown, n_ids, n_mapped = convert_example(
                ex=ex,
                id2label=id2label,
                prompt_text=args.prompt_text,
                unknown_policy=args.unknown_policy,
            )
            fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
            total_examples += 1
            total_ids += n_ids
            mapped_ids += n_mapped
            unknown_pool.extend(unknown)

    unknown_unique = sorted(set(unknown_pool))
    print(f"[DONE] Wrote: {out_path}")
    print(f"[STAT] examples={total_examples}")
    print(f"[STAT] ids_total={total_ids} mapped={mapped_ids}")
    print(
        f"[STAT] unknown_ids={len(unknown_pool)} unique={len(unknown_unique)} "
        f"policy={args.unknown_policy}"
    )
    if unknown_unique:
        print("[SAMPLE_UNKNOWN]", ", ".join(unknown_unique[:20]))


if __name__ == "__main__":
    main()
