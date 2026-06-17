#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate existing ODIL prediction files with strict and soft metrics.

This script does not train or modify prediction files. It discovers existing
result files, normalizes their different schemas, computes per-image exact
match and hierarchical soft-F1, then writes CSV/Markdown reports.

Soft metric:
- If a local metrics_soft.soft_prf is available, it is imported and used.
- Otherwise the fallback follows V6_TRAINING_LOGIC.md:
  sim(a,b) = depth(LCA(a,b)) / max(depth(a), depth(b))
  with root depth 0, then per-example soft P/R/F1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


TERM_SPLIT_RE = re.compile(r"[,;\n\r]+")
TIMEL_ID_RE = re.compile(r"\btm-[a-z0-9]+\b", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


try:
    from metrics_soft import soft_prf as IMPORTED_SOFT_PRF  # type: ignore
except Exception:
    IMPORTED_SOFT_PRF = None


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def normalize_term_key(text: str) -> str:
    """Normalize labels the same way as the V6 training script."""
    if not isinstance(text, str):
        return ""
    text = text.strip().replace("’", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\"'“”‘’.,;:!?-]+", "", text)
    text = re.sub(r"[\s\"'“”‘’.,;:!?-]+$", "", text)
    return text


def image_id_from_path(path_value: str) -> str:
    """Use basename as stable image_id across local and remote absolute paths."""
    if not path_value:
        return ""
    return Path(str(path_value).strip()).name


def first_image_path(obj: Dict[str, Any]) -> str:
    for key in ("image", "image_path", "path"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    images = obj.get("images")
    if isinstance(images, list) and images:
        return str(images[0]).strip()
    return ""


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                warn(f"{path}:{line_no} invalid JSON skipped: {exc}")
                continue
            if isinstance(obj, dict):
                yield line_no, obj
            else:
                warn(f"{path}:{line_no} non-object JSON skipped")


def read_delimited(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row_no, row in enumerate(reader, start=2):
            yield row_no, dict(row)


def load_classes(classes_tsv: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    normalized_to_canonical: Dict[str, str] = {}
    canonical_to_id: Dict[str, str] = {}
    id_to_label: Dict[str, str] = {}

    with classes_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"timel_id", "timel_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{classes_tsv} missing columns {sorted(required)}")
        for row in reader:
            tid = (row.get("timel_id") or "").strip()
            label = (row.get("timel_label") or "").strip()
            if not tid or not label:
                continue
            id_to_label[tid] = label
            canonical_to_id[label] = tid
            normalized_to_canonical[normalize_term_key(label)] = label

    if not id_to_label:
        raise ValueError(f"No TIMEL classes loaded from {classes_tsv}")
    return normalized_to_canonical, canonical_to_id, id_to_label


def load_timel_taxonomy(
    taxonomy_json: Path,
    id_to_label: Dict[str, str],
) -> Dict[str, List[str]]:
    """Return leaf_id -> path_ids, following the V6 loader behavior."""
    data = json.loads(taxonomy_json.read_text(encoding="utf-8"))

    id_to_value: Dict[str, str] = {}
    for item in data.get("items", []):
        path = item.get("path_ids") or []
        if path:
            leaf = str(path[-1])
            id_to_value.setdefault(leaf, (item.get("value") or "").strip())

    id_to_path_ids: Dict[str, List[str]] = {}
    for item in data.get("items", []):
        path = [str(x) for x in (item.get("path_ids") or []) if str(x)]
        if not path:
            continue
        leaf = path[-1]
        if leaf in id_to_path_ids:
            continue
        # Keep all taxonomy nodes in the path; leaf filtering happens later.
        id_to_path_ids[leaf] = path

    missing = [tid for tid in id_to_label if tid not in id_to_path_ids]
    if missing:
        warn(f"{len(missing)} class IDs are absent from taxonomy paths; their soft score falls back to exact match.")
    return id_to_path_ids


def fallback_soft_prf(
    pred_ids: Sequence[str],
    gold_ids: Sequence[str],
    id_to_path_ids: Dict[str, List[str]],
) -> Tuple[float, float, float]:
    """V6 documented soft P/R/F1 over multi-label predictions."""
    pred = unique_ordered(pred_ids)
    gold = unique_ordered(gold_ids)

    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0

    def sim(a: str, b: str) -> float:
        if a == b:
            return 1.0
        pa = id_to_path_ids.get(a)
        pb = id_to_path_ids.get(b)
        if not pa or not pb:
            return 0.0
        common_depth = -1
        for depth, (xa, xb) in enumerate(zip(pa, pb)):
            if xa != xb:
                break
            common_depth = depth
        if common_depth <= 0:
            return 0.0
        denom = max(len(pa) - 1, len(pb) - 1)
        return (common_depth / denom) if denom > 0 else 0.0

    precision = sum(max(sim(p, g) for g in gold) for p in pred) / len(pred)
    recall = sum(max(sim(g, p) for p in pred) for g in gold) / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def unique_ordered(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def flatten_value(value: Any) -> List[str]:
    """Convert lists, JSON-ish strings, and free text to candidate labels/IDs."""
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(flatten_value(item))
        return out
    if isinstance(value, tuple) or isinstance(value, set):
        out = []
        for item in value:
            out.extend(flatten_value(item))
        return out
    if not isinstance(value, str):
        return [str(value)]

    text = value.strip()
    if not text:
        return []

    # JSON or Python-list-looking cells from notebooks / CSVs.
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        try:
            parsed = json.loads(text)
            return flatten_value(parsed)
        except Exception:
            pass

    ids = TIMEL_ID_RE.findall(text)
    if ids and len(ids) == len(TIMEL_ID_RE.findall(text.replace(",", " "))):
        # Preserve text splitting below when IDs appear together with labels.
        return ids

    # For path-based predictions, keep only the leaf segment per path chunk.
    if " / " in text:
        chunks = [chunk.strip() for chunk in re.split(r"\s*;\s*", text) if chunk.strip()]
        leaves = []
        for chunk in chunks:
            segments = [seg.strip() for seg in chunk.split(" / ") if seg.strip()]
            if segments:
                leaves.append(segments[-1])
        if leaves:
            return leaves

    return [piece.strip() for piece in TERM_SPLIT_RE.split(text) if piece.strip()]


def candidates_to_ids(
    value: Any,
    normalized_to_canonical: Dict[str, str],
    canonical_to_id: Dict[str, str],
    id_to_label: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (ids, canonical_labels, unknown_candidates)."""
    ids: List[str] = []
    labels: List[str] = []
    unknown: List[str] = []

    for candidate in flatten_value(value):
        candidate = str(candidate).strip()
        if not candidate:
            continue
        if TIMEL_ID_RE.fullmatch(candidate):
            tid = candidate
            if tid in id_to_label:
                ids.append(tid)
                labels.append(id_to_label[tid])
            else:
                unknown.append(candidate)
            continue
        key = normalize_term_key(candidate)
        canonical = normalized_to_canonical.get(key)
        if canonical and canonical in canonical_to_id:
            labels.append(canonical)
            ids.append(canonical_to_id[canonical])
        else:
            unknown.append(candidate)

    ids = unique_ordered(ids)
    labels = [id_to_label[tid] for tid in ids if tid in id_to_label]
    return ids, labels, unique_ordered(unknown)


def assistant_text_from_sft_record(obj: Dict[str, Any]) -> str:
    for msg in obj.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            if parts:
                return " ".join(parts)
    return ""


def load_gold_maps(
    gold_jsonls: Sequence[Path],
    normalized_to_canonical: Dict[str, str],
    canonical_to_id: Dict[str, str],
    id_to_label: Dict[str, str],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Build image_id -> (gold_ids, gold_labels) from annotated JSONL files."""
    gold_by_image: Dict[str, Tuple[List[str], List[str]]] = {}
    for path in gold_jsonls:
        if not path.exists():
            warn(f"Gold JSONL not found: {path}")
            continue
        for _, obj in read_jsonl(path):
            image_path = first_image_path(obj)
            image_id = image_id_from_path(image_path)
            if not image_id:
                continue
            text = assistant_text_from_sft_record(obj)
            ids, labels, _unknown = candidates_to_ids(
                text, normalized_to_canonical, canonical_to_id, id_to_label
            )
            if ids:
                gold_by_image[image_id] = (ids, labels)
    return gold_by_image


def pick_first_present(obj: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return None


def pick_prediction_value(obj: Dict[str, Any]) -> Tuple[bool, Any]:
    """
    Pick a prediction field.

    Empty string predictions are valid outputs and should score 0, not be
    skipped. Empty normalized lists from old runs are less reliable, so fall
    back to raw_prediction when available.
    """
    for key in ("predicted_ids", "predicted_terms", "predicted_label", "predicted_labels", "prediction"):
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, list) and len(value) == 0:
            continue
        if value not in (None, ""):
            return True, value
    if "pred" in obj:
        return True, obj.get("pred", "")
    if "raw_prediction" in obj:
        return True, obj.get("raw_prediction", "")
    return False, None


def standardize_record(
    obj: Dict[str, Any],
    source_file: Path,
    line_no: int,
    experiment_name: str,
    gold_by_image: Dict[str, Tuple[List[str], List[str]]],
    normalized_to_canonical: Dict[str, str],
    canonical_to_id: Dict[str, str],
    id_to_label: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    image_path = first_image_path(obj)
    image_id = image_id_from_path(image_path)
    if not image_id:
        warn(f"{source_file}:{line_no} missing image/image_path; skipped")
        return None

    has_pred, pred_value = pick_prediction_value(obj)
    if not has_pred:
        warn(f"{source_file}:{line_no} missing predicted label field; skipped")
        return None

    gold_value = pick_first_present(
        obj,
        (
            "gold_ids",
            "gold_terms",
            "true_label",
            "true_labels",
            "ground_truth_label",
            "ground_truth_labels",
            "ground_truth",
            "gt",
            "label",
            "labels",
        ),
    )

    pred_ids, pred_labels, pred_unknown = candidates_to_ids(
        pred_value, normalized_to_canonical, canonical_to_id, id_to_label
    )

    if gold_value is not None:
        gold_ids, gold_labels, _gold_unknown = candidates_to_ids(
            gold_value, normalized_to_canonical, canonical_to_id, id_to_label
        )
    else:
        gold_ids, gold_labels = gold_by_image.get(image_id, ([], []))

    if not gold_ids:
        warn(f"{source_file}:{line_no} missing or unmappable gold labels for {image_id}; skipped")
        return None

    return {
        "image_id": image_id,
        "image_path": image_path,
        "true_label": "; ".join(gold_labels),
        "predicted_label": "; ".join(pred_labels),
        "true_label_ids": "; ".join(gold_ids),
        "predicted_label_ids": "; ".join(pred_ids),
        "experiment_name": experiment_name,
        "source_file": str(source_file),
        "predicted_unknown": "; ".join(pred_unknown),
        "_true_ids": gold_ids,
        "_pred_ids": pred_ids,
        "_has_unknown_pred": bool(pred_unknown),
    }


def result_like_jsonl(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".jsonl", ".csv", ".tsv"}:
        return False
    if "per_class" in name:
        return False
    return any(token in name for token in ("result", "pred"))


def discover_result_files(results_dir: Path, project_root: Path, include_root_results: bool) -> List[Path]:
    files: List[Path] = []
    if results_dir.exists():
        files.extend(p for p in results_dir.rglob("*") if p.is_file() and result_like_jsonl(p))
    else:
        warn(f"results_dir does not exist: {results_dir}")

    if include_root_results:
        files.extend(
            p for p in project_root.glob("*")
            if p.is_file() and result_like_jsonl(p)
        )

    # Stable order and de-duplication by resolved path.
    seen: Set[Path] = set()
    out: List[Path] = []
    for path in sorted(files):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def experiment_name_for(path: Path, results_dir: Path, project_root: Path) -> str:
    try:
        rel = path.relative_to(results_dir)
        parts = list(rel.parts)
        if len(parts) >= 2:
            return f"{parts[0]}__{path.stem}"
        return path.stem
    except ValueError:
        try:
            rel_root = path.relative_to(project_root)
            return f"root__{rel_root.stem}"
        except ValueError:
            return path.stem


def read_prediction_file(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from read_jsonl(path)
    elif suffix in {".csv", ".tsv"}:
        yield from read_delimited(path)


def evaluate_records(
    records: List[Dict[str, Any]],
    soft_fn: Callable[[Sequence[str], Sequence[str], Dict[str, List[str]]], Tuple[float, float, float]],
    id_to_path_ids: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []
    for rec in records:
        pred_ids = rec.pop("_pred_ids")
        true_ids = rec.pop("_true_ids")
        has_unknown_pred = rec.pop("_has_unknown_pred")
        strict = 1.0 if (set(pred_ids) == set(true_ids) and not has_unknown_pred) else 0.0
        _sp, _sr, soft_f1 = soft_fn(pred_ids, true_ids, id_to_path_ids)
        rec["strict_score"] = strict
        rec["soft_score"] = soft_f1
        evaluated.append(rec)
    return evaluated


def summarize(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[rec["experiment_name"]].append(rec)

    summary: List[Dict[str, Any]] = []
    for name in sorted(groups):
        rows = groups[name]
        n = len(rows)
        mean_strict = sum(float(r["strict_score"]) for r in rows) / n if n else 0.0
        mean_soft = sum(float(r["soft_score"]) for r in rows) / n if n else 0.0
        summary.append({
            "experiment_name": name,
            "mean_strict_score": mean_strict,
            "mean_soft_score": mean_soft,
            "sample_count": n,
        })
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_md(path: Path, summary: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "| experiment_name | mean_strict_score | mean_soft_score | sample_count |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['experiment_name']} | "
            f"{float(row['mean_strict_score']):.6f} | "
            f"{float(row['mean_soft_score']):.6f} | "
            f"{int(row['sample_count'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: Sequence[Dict[str, Any]]) -> None:
    if not summary:
        print("No valid experiment records found.")
        return
    headers = ("experiment_name", "strict", "soft", "n")
    name_w = max(len(headers[0]), *(len(str(r["experiment_name"])) for r in summary))
    print()
    print(f"{headers[0]:<{name_w}}  {headers[1]:>10}  {headers[2]:>10}  {headers[3]:>6}")
    print(f"{'-' * name_w}  {'-' * 10}  {'-' * 10}  {'-' * 6}")
    for row in summary:
        print(
            f"{row['experiment_name']:<{name_w}}  "
            f"{float(row['mean_strict_score']):>10.6f}  "
            f"{float(row['mean_soft_score']):>10.6f}  "
            f"{int(row['sample_count']):>6}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("evaluation_outputs"))
    p.add_argument("--project_root", type=Path, default=Path("."))
    p.add_argument("--classes_tsv", type=Path, default=Path("02_source_data/data/classes.tsv"))
    p.add_argument("--taxonomy_json", type=Path, default=Path("02_source_data/data/timel_taxonomy.json"))
    p.add_argument(
        "--gold_jsonl",
        type=Path,
        action="append",
        default=[
            Path("finetunning/03_full_10000_training/odil_10000/val.jsonl"),
            Path("finetunning/03_full_10000_training/odil_10000/val_terms.jsonl"),
        ],
        help="Annotated JSONL used to fill missing gold labels; may be repeated.",
    )
    p.add_argument(
        "--include_root_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also include result-like files in project root, e.g. results-10000-ID_v4.jsonl.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = (project_root / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    classes_tsv = (project_root / args.classes_tsv).resolve() if not args.classes_tsv.is_absolute() else args.classes_tsv
    taxonomy_json = (project_root / args.taxonomy_json).resolve() if not args.taxonomy_json.is_absolute() else args.taxonomy_json
    gold_jsonls = [
        (project_root / p).resolve() if not p.is_absolute() else p
        for p in args.gold_jsonl
    ]

    normalized_to_canonical, canonical_to_id, id_to_label = load_classes(classes_tsv)
    id_to_path_ids = load_timel_taxonomy(taxonomy_json, id_to_label)
    gold_by_image = load_gold_maps(gold_jsonls, normalized_to_canonical, canonical_to_id, id_to_label)
    print(f"[INFO] Loaded {len(id_to_label)} TIMEL labels")
    print(f"[INFO] Loaded taxonomy paths for {len(id_to_path_ids)} IDs")
    print(f"[INFO] Loaded fallback gold labels for {len(gold_by_image)} images")

    soft_fn = IMPORTED_SOFT_PRF or fallback_soft_prf
    print(f"[INFO] Soft metric source: {'metrics_soft.soft_prf' if IMPORTED_SOFT_PRF else 'V6_TRAINING_LOGIC.md fallback'}")

    files = discover_result_files(results_dir, project_root, args.include_root_results)
    if not files:
        warn("No result-like files found.")

    all_records: List[Dict[str, Any]] = []
    for path in files:
        experiment_name = experiment_name_for(path, results_dir, project_root)
        usable = 0
        skipped = 0
        for line_no, obj in read_prediction_file(path):
            rec = standardize_record(
                obj=obj,
                source_file=path,
                line_no=line_no,
                experiment_name=experiment_name,
                gold_by_image=gold_by_image,
                normalized_to_canonical=normalized_to_canonical,
                canonical_to_id=canonical_to_id,
                id_to_label=id_to_label,
            )
            if rec is None:
                skipped += 1
                continue
            all_records.append(rec)
            usable += 1
        if usable == 0:
            warn(f"{path} produced no usable records and was skipped as an experiment")
        else:
            print(f"[INFO] {experiment_name}: usable={usable} skipped={skipped} source={path}")

    evaluated = evaluate_records(all_records, soft_fn, id_to_path_ids)
    summary = summarize(evaluated)

    by_prediction_fields = [
        "image_id",
        "image_path",
        "true_label",
        "predicted_label",
        "experiment_name",
        "strict_score",
        "soft_score",
        "true_label_ids",
        "predicted_label_ids",
        "predicted_unknown",
        "source_file",
    ]
    summary_fields = [
        "experiment_name",
        "mean_strict_score",
        "mean_soft_score",
        "sample_count",
    ]

    write_csv(output_dir / "metrics_by_prediction.csv", evaluated, by_prediction_fields)
    write_csv(output_dir / "metrics_summary.csv", summary, summary_fields)
    write_summary_md(output_dir / "metrics_summary.md", summary)

    print_summary(summary)
    print()
    print(f"[DONE] Wrote {output_dir / 'metrics_by_prediction.csv'}")
    print(f"[DONE] Wrote {output_dir / 'metrics_summary.csv'}")
    print(f"[DONE] Wrote {output_dir / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
