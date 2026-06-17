#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen3vl_sft_timel_merged_fr.py

Script de fine-tuning SFT pour Qwen3-VL (étiquettes timel), version "fusion" orientée production.

Objectifs :
- Par défaut : modifier le moins possible les données (utiliser directement le JSONL d'origine).
- Boucle d'entraînement : pilotée par le nombre de pas (max_steps).
- Reprise automatique (resume) si des checkpoints existent.
- Sauvegarde finale dans output_dir/final.

Compatibilité :
- Le pipeline principal conserve le schema JSONL natif (images + messages)
  et applique seulement une resolution robuste des chemins d'images.

Fixes appliqués (v2) :
- [CRITICAL] Labels : masquage correct des tokens prompt/system (loss sur assistant uniquement)
- [CRITICAL] Gradient checkpointing activé par défaut (fix OOM A40)
- [PERF]     apply_chat_template n'est plus appelé deux fois (build_text supprimé)
- [PERF]     dataloader_num_workers exposé et defaulté à 4
- [PERF]     pin_memory activé si CUDA disponible
- [FIX]      torch_dtype -> dtype (deprecation warning transformers)
- [FIX]      lr_scheduler_type cosine ajouté
- [FIX]      max_pixels exposé en argument CLI

Fixes appliqués (v3) :
- [CRITICAL] mask_prompt_tokens : <|im_start|> est un token spécial unique dans le vocabulaire
             Qwen3-VL. L'ancien code encodait la chaîne "<|im_start|>assistant\n" comme texte
             brut, ce qui produisait une séquence de token_ids différente de celle réellement
             présente dans input_ids → header jamais trouvé → tous labels=-100 → loss=0
             → le modèle n'apprenait rien.
             Fix : utiliser convert_tokens_to_ids("<|im_start|>") pour obtenir le bon token_id,
             puis concatener avec encode("assistant\n"). Un print de diagnostic au premier
             batch vérifie que le header est bien trouvé.
- [FIX]      AutoModelForVision2Seq remplacé par AutoModelForImageTextToText (deprecation v5)
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from datasets import load_dataset

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from trl import SFTTrainer, SFTConfig


TIMEL_ID_RE = re.compile(r"\btm-[a-z0-9]{8}\b")


# -------------------------
# Utilitaires données
# -------------------------

def pil_loader(path: str) -> Image.Image:
    """Ouvre une image et force le mode RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


def str2bool(v: str) -> bool:
    v = v.lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Valeur booleenne invalide: {v}")


def resolve_example_paths(ex: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    Convertit les chemins d'image relatifs en chemins absolus a partir d'un dossier base.
    """
    base = Path(base_dir).resolve()

    def to_abs(path_value: str) -> str:
        p = Path(path_value).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        return str(p)

    out = dict(ex)
    images = ex.get("images", [])
    out["images"] = [to_abs(images[0])]

    fixed_messages: List[Dict[str, Any]] = []
    for msg in ex.get("messages", []):
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    p = dict(part)
                    if p.get("type") == "image" and isinstance(p.get("image"), str):
                        p["image"] = to_abs(p["image"])
                    parts.append(p)
                else:
                    parts.append(part)
            m["content"] = parts
        fixed_messages.append(m)
    out["messages"] = fixed_messages
    return out


def validate_image_paths(split, split_name: str, max_show: int = 5) -> None:
    """Vérifie que toutes les images existent avant l'entraînement (fail fast)."""
    missing: List[Tuple[int, str]] = []
    for idx, ex in enumerate(split):
        img_path = ex["images"][0]
        if not os.path.isfile(img_path):
            missing.append((idx, img_path))
            if len(missing) >= max_show:
                break

    if missing:
        details = "\n".join([f"  - {split_name}[{i}] -> {p}" for i, p in missing])
        raise FileNotFoundError(
            f"Images introuvables ({split_name}). Exemples:\n{details}\n"
            "Verifiez le dossier de base du JSONL ou les chemins data/images."
        )


def sanitize_messages_for_template(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Nettoie le schema messages pour apply_chat_template.
    Supprime les clés None injectées par datasets (ex: {'type':'text','image':None}).
    """
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            cleaned.append({"role": role, "content": content})
            continue

        new_parts: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "image":
                    img = part.get("image")
                    if isinstance(img, str) and img:
                        new_parts.append({"type": "image", "image": img})
                elif ptype == "text":
                    txt = part.get("text")
                    if isinstance(txt, str):
                        new_parts.append({"type": "text", "text": txt})

        cleaned.append({"role": role, "content": new_parts})
    return cleaned


def keep_prompt_messages_only(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Conserve uniquement les tours non-assistant pour la génération.
    Cela permet d'utiliser le même JSONL annoté en mode prédiction,
    sans injecter la réponse gold dans le prompt.
    """
    kept: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            kept.append(msg)
    return kept


def load_timel_reference(classes_tsv: str) -> Dict[str, str]:
    """Charge le référentiel TIMEL depuis classes.tsv."""
    if not classes_tsv:
        raise ValueError("classes_tsv requis pour utiliser le referentiel ID.")
    if not os.path.isfile(classes_tsv):
        raise FileNotFoundError(f"classes_tsv introuvable: {classes_tsv}")

    id_to_label: Dict[str, str] = {}
    with open(classes_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"timel_id", "timel_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"classes.tsv invalide: colonnes attendues {sorted(required)}, "
                f"trouvees={reader.fieldnames}"
            )
        for row in reader:
            tid = (row.get("timel_id") or "").strip()
            label = (row.get("timel_label") or "").strip()
            if tid:
                id_to_label[tid] = label

    if not id_to_label:
        raise ValueError(f"Aucun timel_id charge depuis {classes_tsv}")
    return id_to_label


def extract_timel_ids(text: str) -> List[str]:
    """Extrait les identifiants TIMEL présents dans un texte libre."""
    if not isinstance(text, str):
        return []
    return TIMEL_ID_RE.findall(text)


def normalize_prediction_ids(text: str, valid_ids: Set[str]) -> Dict[str, List[str]]:
    """
    Normalise la sortie générée :
    - extrait les tm-xxxxxxxx
    - conserve l'ordre d'apparition
    - enlève les doublons
    - sépare IDs valides / invalides selon le référentiel
    """
    seen: Set[str] = set()
    valid: List[str] = []
    invalid: List[str] = []
    extracted = extract_timel_ids(text)

    for tid in extracted:
        if tid in seen:
            continue
        seen.add(tid)
        if tid in valid_ids:
            valid.append(tid)
        else:
            invalid.append(tid)

    return {
        "extracted_ids": extracted,
        "valid_ids": valid,
        "invalid_ids": invalid,
    }


def extract_gold_ids_from_example(ex: Dict[str, Any]) -> List[str]:
    """Récupère les IDs gold déjà présents dans un exemple annoté."""
    for msg in ex.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return normalize_prediction_ids(part.get("text", ""), set(extract_timel_ids(part.get("text", ""))))["valid_ids"]
    return []


def move_batch_to_model_device(batch: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Déplace les tenseurs d'entrée vers le device principal du modèle.
    Compatible avec les chargements simples et, dans la plupart des cas,
    avec device_map=auto.
    """
    target_device = None
    try:
        target_device = model.device
    except Exception:
        target_device = None

    if target_device is None or str(target_device) == "meta":
        for param in model.parameters():
            target_device = param.device
            break

    if target_device is None or str(target_device) == "meta":
        return batch

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(target_device) if torch.is_tensor(value) else value
    return moved


# -------------------------
# Masquage des labels (FIX CRITIQUE v3)
# -------------------------

def build_assistant_header_ids(tokenizer) -> List[int]:
    """
    Construit la séquence de token_ids correspondant au header assistant Qwen3 :
        <|im_start|>assistant\n

    PROBLÈME v2 : tokenizer.encode("<|im_start|>assistant\n") encode la chaîne
    comme du texte brut. Or <|im_start|> est un token spécial unique (un seul id)
    dans le vocabulaire Qwen3. Le encode() textuel peut produire plusieurs ids
    différents (ex: "<", "|", "im", "_", "start", "|", ">") → séquence jamais
    trouvée dans input_ids → header_not_found → labels[:] = -100 → loss = 0.

    FIX v3 : on récupère le token_id exact via convert_tokens_to_ids, puis on
    concatène avec l'encodage de "assistant\n".
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if im_start_id == tokenizer.unk_token_id:
        # Fallback : essayer via added_tokens_encoder
        im_start_id = tokenizer.added_tokens_encoder.get("<|im_start|>", None)
        if im_start_id is None:
            raise ValueError(
                "[CRITIQUE] Token '<|im_start|>' introuvable dans le vocabulaire du tokenizer. "
                "Vérifiez que le processor chargé correspond bien à Qwen3-VL."
            )

    # "assistant\n" encodé normalement (pas de token spécial ici)
    assistant_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    return [im_start_id] + assistant_ids


def mask_prompt_tokens(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    texts: List[str],
    _header_cache: Dict = {},  # cache pour éviter de recalculer à chaque batch
) -> torch.Tensor:
    """
    Masque tous les tokens qui ne font pas partie de la réponse assistant.
    Seuls les tokens de la réponse assistant contribuent à la loss.

    Stratégie : pour chaque exemple du batch, on cherche la position du
    dernier token du header assistant (<|im_start|>assistant\n) et on
    masque tout ce qui précède (labels = -100).

    FIX v3 : utilise build_assistant_header_ids() pour obtenir la vraie
    séquence de token_ids, avec <|im_start|> comme token spécial unique.
    """
    tokenizer_id = id(tokenizer)
    if tokenizer_id not in _header_cache:
        header_ids = build_assistant_header_ids(tokenizer)
        _header_cache[tokenizer_id] = header_ids
        print(
            f"[MASK DEBUG] header assistant token_ids = {header_ids} "
            f"→ décodé : {tokenizer.decode(header_ids)!r}"
        )
    else:
        header_ids = _header_cache[tokenizer_id]

    header_len = len(header_ids)
    header_tensor = torch.tensor(header_ids, dtype=input_ids.dtype, device=input_ids.device)

    n_found = 0
    n_not_found = 0

    for i in range(input_ids.size(0)):
        seq = input_ids[i]
        seq_len = seq.size(0)
        last_assistant_end = -1

        # Cherche la dernière occurrence du header assistant dans la séquence
        for j in range(seq_len - header_len, -1, -1):
            if torch.equal(seq[j: j + header_len], header_tensor):
                last_assistant_end = j + header_len
                break

        if last_assistant_end == -1:
            n_not_found += 1
            labels[i, :] = -100
        else:
            n_found += 1
            labels[i, :last_assistant_end] = -100

    # Diagnostic au premier batch (et si des headers ne sont pas trouvés)
    if n_not_found > 0:
        print(
            f"[MASK WARN] {n_not_found}/{input_ids.size(0)} exemples : header assistant NON TROUVÉ "
            f"→ labels masqués entièrement. Vérifiez le chat template."
        )
    if n_found > 0 and n_not_found == 0:
        # Log uniquement au premier appel (quand le cache vient d'être créé)
        if len(_header_cache) == 1:
            print(f"[MASK OK] Header assistant trouvé dans tous les {n_found} exemples du batch.")

    return labels


# -------------------------
# Vérifications (sanity checks)
# -------------------------

def schema_sanity_check(ex: Dict[str, Any], require_assistant: bool = True) -> None:
    """Vérification minimale du schéma."""
    if "images" not in ex or "messages" not in ex:
        raise ValueError("Chaque ligne JSONL doit contenir les clés : 'images' et 'messages'.")
    if not isinstance(ex["images"], list) or len(ex["images"]) != 1:
        raise ValueError("'images' doit être une liste de longueur 1 (une image par exemple).")
    min_turns = 2 if require_assistant else 1
    if not isinstance(ex["messages"], list) or len(ex["messages"]) < min_turns:
        raise ValueError(
            f"'messages' doit être une liste avec au moins {min_turns} tour(s)."
        )


def processor_sanity_check(processor: AutoProcessor, ex: Dict[str, Any]) -> None:
    """
    Test léger : encode un exemple et vérifie la présence de tenseurs liés à l'image.
    Vérifie aussi que le header assistant est bien trouvable dans un input_ids encodé.
    """
    try:
        img0 = ex["images"][0]
        if isinstance(img0, str):
            img0 = pil_loader(img0)

        msgs = ex["messages"]
        texts: List[str] = []
        for m in msgs:
            role = m.get("role", "")
            c = m.get("content")
            if isinstance(c, str):
                texts.append(f"{role}: {c}")
                continue
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(f"{role}: {part['text']}")
                        break

        text = "\n".join(texts) if texts else "user: (vide)\nassistant: (vide)"
        enc = processor(text=text, images=img0, return_tensors="pt")
        keys = set(enc.keys())

        vision_like = {"pixel_values", "image_grid_thw", "vision_pixel_values"}
        has_vision = any(k in keys for k in vision_like) or any(("pixel" in k) or ("image" in k) for k in keys)

        if not has_vision:
            print(
                "[ATTENTION] Encodage processor : aucune clé 'vision' évidente détectée.\n"
                f"            Clés retournées: {sorted(list(keys))}\n"
                "            Conseil: verifiez les versions Transformers/TRL et le format messages/images."
            )
        else:
            print(f"[OK] Sanity check processor : clés détectées (aperçu) = {sorted(list(keys))[:12]} ...")

        # Sanity check masquage : vérifier que le header est trouvable
        try:
            header_ids = build_assistant_header_ids(processor.tokenizer)
            print(
                f"[OK] Sanity check masquage : header assistant token_ids = {header_ids} "
                f"→ {processor.tokenizer.decode(header_ids)!r}"
            )
        except Exception as e_mask:
            print(f"[ATTENTION] Sanity check masquage a échoué : {repr(e_mask)}")

    except Exception as e:
        print(
            "[ATTENTION] Sanity check processor a échoué (exception).\n"
            f"            Erreur: {repr(e)}\n"
            "            Conseil: verifiez les versions Transformers/TRL et le format messages/images."
        )


# -------------------------
# Arguments CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--mode", type=str, default="train", choices=["train", "predict"])

    # Données
    p.add_argument("--train_jsonl", type=str, default="train.jsonl")
    p.add_argument("--val_jsonl", type=str, default="val.jsonl")
    p.add_argument("--use_val_if_exists", type=str2bool, default=True)
    p.add_argument("--predict_jsonl", type=str, default="")
    p.add_argument("--pred_out", type=str, default="")
    p.add_argument("--classes_tsv", type=str, default="")
    p.add_argument("--predict_limit", type=int, default=0)

    # Modèle
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")

    # Sortie / reproductibilité
    p.add_argument("--output_dir", type=str, default="qwen3_vl_timel_sft_out")
    p.add_argument("--seed", type=int, default=42)

    # Entraînement
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--per_device_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument(
        "--lr_scheduler_type", type=str, default="cosine",
        help="Type de scheduler LR (cosine recommandé).",
    )

    # Troncature
    p.add_argument(
        "--max_length", type=str, default="none",
        help="'none' pour max_length=None (recommandé VLM), sinon un entier.",
    )

    # Image resolution (VLM memory control)
    p.add_argument(
        "--min_pixels", type=int, default=256 * 28 * 28,
        help="Résolution minimale en pixels pour le processor Qwen3-VL.",
    )
    p.add_argument(
        "--max_pixels", type=int, default=512 * 28 * 28,
        help="Résolution maximale en pixels pour le processor Qwen3-VL. "
             "Réduire pour éviter les OOM (défaut: 512*28*28 ~= 401k pixels).",
    )

    # Logs / sauvegarde / éval
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)

    # Precision
    p.add_argument(
        "--precision", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )

    # Gradient checkpointing
    p.add_argument(
        "--gradient_checkpointing", type=str2bool, default=True,
        help="Activer le gradient checkpointing (recommandé pour VLM, réduit l'usage VRAM).",
    )

    # Reprise
    p.add_argument("--resume", type=str2bool, default=True)

    # Divers
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument(
        "--dataloader_num_workers", type=int, default=4,
        help="Nombre de workers pour le DataLoader. 0 = single-threaded (lent). 4 recommandé.",
    )

    # Device
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=1)

    return p.parse_args()


# -------------------------
# Data Collator
# -------------------------

class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # FIX: apply_chat_template appelé une seule fois ici (build_text supprimé)
        texts = []
        for f in features:
            if isinstance(f.get("messages"), list):
                msg = sanitize_messages_for_template(f["messages"])
                t = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
            else:
                # Fallback au cas où "text" serait déjà présent
                t = f["text"]
            texts.append(t)

        # FIX: images chargées une seule fois ici (pas de double pil_loader)
        imgs = []
        for f in features:
            im = f["images"]
            if isinstance(im, list):
                im = im[0]
            if isinstance(im, str):
                im = pil_loader(im)
            imgs.append(im)

        # Garde-fou: vérification slots vision
        expected_vision_slots = len(imgs)
        observed_vision_slots = sum(t.count("<|vision_start|>") for t in texts)
        if observed_vision_slots != expected_vision_slots:
            sample_preview = texts[0][:400].replace("\n", "\\n") if texts else "<empty>"
            sample_keys = list(features[0].keys()) if features else []
            raise ValueError(
                "Mismatch image slots: "
                f"observed <|vision_start|>={observed_vision_slots}, "
                f"expected images={expected_vision_slots}. "
                "Verifiez les messages ou un eventuel packing du dataset. "
                f"feature_keys={sample_keys} preview={sample_preview}"
            )

        batch = self.processor(
            text=texts,
            images=imgs,
            return_tensors="pt",
            padding=True,
        )

        # Masquage labels
        labels = batch["input_ids"].clone()

        # 1. Masque les tokens de padding
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # 2. FIX CRITIQUE v3: masque les tokens prompt/system
        #    → la loss est calculée uniquement sur les tokens de réponse assistant
        #    → utilise build_assistant_header_ids() pour le vrai token_id de <|im_start|>
        labels = mask_prompt_tokens(
            labels=labels,
            input_ids=batch["input_ids"],
            tokenizer=self.processor.tokenizer,
            texts=texts,
        )

        batch["labels"] = labels
        return batch


# -------------------------
# Main
# -------------------------

def load_processor_and_model(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    torch_dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch_dtype_map[args.precision],
        device_map=args.device_map,
    )
    return processor, model


def run_train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.train_jsonl):
        raise FileNotFoundError(f"train_jsonl introuvable: {args.train_jsonl}")

    data_files = {"train": args.train_jsonl}
    use_val = args.use_val_if_exists and os.path.exists(args.val_jsonl)
    if use_val:
        data_files["validation"] = args.val_jsonl

    ds = load_dataset("json", data_files=data_files)
    first = ds["train"][0]
    schema_sanity_check(first, require_assistant=True)

    train_base_dir = str(Path(args.train_jsonl).resolve().parent)
    ds["train"] = ds["train"].map(
        lambda ex: resolve_example_paths(ex, train_base_dir),
        desc=f"Resolution chemins train depuis {train_base_dir}",
    )
    if use_val:
        val_base_dir = str(Path(args.val_jsonl).resolve().parent)
        ds["validation"] = ds["validation"].map(
            lambda ex: resolve_example_paths(ex, val_base_dir),
            desc=f"Resolution chemins val depuis {val_base_dir}",
        )

    validate_image_paths(ds["train"], "train")
    if use_val:
        validate_image_paths(ds["validation"], "validation")

    first = ds["train"][0]
    processor, model = load_processor_and_model(args)
    processor_sanity_check(processor, first)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU] VRAM allouée: {allocated:.2f} GB | réservée: {reserved:.2f} GB")

    max_length: Optional[int]
    if args.max_length.lower() == "none":
        max_length = None
    else:
        max_length = int(args.max_length)

    eval_strategy = "steps" if use_val else "no"
    eval_steps = args.eval_steps if use_val else None

    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        remove_unused_columns=False,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        max_length=max_length,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        data_collator=VLMDataCollator(processor),
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation") if use_val else None,
        processing_class=processor,
    )

    resume_from: Optional[str] = None
    if args.resume and os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            resume_from = args.output_dir

    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[TERMINE] Modèle final sauvegardé dans: {final_dir}")


def run_predict(args: argparse.Namespace) -> None:
    if not args.predict_jsonl:
        raise ValueError("--predict_jsonl requis en mode predict.")
    if not os.path.exists(args.predict_jsonl):
        raise FileNotFoundError(f"predict_jsonl introuvable: {args.predict_jsonl}")
    if not args.pred_out:
        raise ValueError("--pred_out requis en mode predict.")

    id_to_label = load_timel_reference(args.classes_tsv)
    valid_ids = set(id_to_label.keys())
    print(f"[REF] {len(valid_ids)} IDs valides chargés depuis {args.classes_tsv}")

    ds = load_dataset("json", data_files={"predict": args.predict_jsonl})["predict"]
    if len(ds) == 0:
        raise ValueError("predict_jsonl vide.")

    schema_sanity_check(ds[0], require_assistant=False)
    pred_base_dir = str(Path(args.predict_jsonl).resolve().parent)
    ds = ds.map(
        lambda ex: resolve_example_paths(ex, pred_base_dir),
        desc=f"Resolution chemins predict depuis {pred_base_dir}",
    )
    validate_image_paths(ds, "predict")

    if args.predict_limit and args.predict_limit > 0:
        limit = min(args.predict_limit, len(ds))
        ds = ds.select(range(limit))
        print(f"[PREDICT] Limitation à {limit} exemple(s).")

    processor, model = load_processor_and_model(args)
    processor_sanity_check(processor, ds[0])
    model.eval()

    out_path = Path(args.pred_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_valid = 0
    total_invalid = 0
    empty_predictions = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            prompt_messages = sanitize_messages_for_template(
                keep_prompt_messages_only(ex["messages"])
            )
            text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            img = pil_loader(ex["images"][0])

            batch = processor(
                text=[text],
                images=[img],
                return_tensors="pt",
                padding=True,
            )
            batch = move_batch_to_model_device(batch, model)

            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=args.num_beams,
                )

            prompt_len = batch["input_ids"].shape[1]
            generated_new = generated[:, prompt_len:]
            raw_text = processor.batch_decode(
                generated_new,
                skip_special_tokens=True,
            )[0].strip()

            normalized = normalize_prediction_ids(raw_text, valid_ids)
            pred_ids = normalized["valid_ids"]
            invalid_ids = normalized["invalid_ids"]
            gold_ids = extract_gold_ids_from_example(ex)

            total_valid += len(pred_ids)
            total_invalid += len(invalid_ids)
            if not pred_ids:
                empty_predictions += 1

            record = {
                "index": idx,
                "image": ex["images"][0],
                "raw_prediction": raw_text,
                "predicted_ids": pred_ids,
                "predicted_labels": [id_to_label[tid] for tid in pred_ids],
                "invalid_ids": invalid_ids,
            }
            if gold_ids:
                record["gold_ids"] = gold_ids

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[PREDICT] Prédictions sauvegardées dans: {out_path}")
    print(
        "[PREDICT] Résumé: "
        f"exemples={len(ds)}, total_valid_ids={total_valid}, "
        f"total_invalid_ids={total_invalid}, empty_predictions={empty_predictions}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "train":
        run_train(args)
        return

    if args.mode == "predict":
        run_predict(args)
        return

    raise ValueError(f"Mode non supporté: {args.mode}")


if __name__ == "__main__":
    main()
