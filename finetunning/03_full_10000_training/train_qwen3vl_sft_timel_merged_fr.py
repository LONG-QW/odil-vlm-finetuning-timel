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
"""


import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from PIL import Image
from datasets import load_dataset


import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, set_seed
from trl import SFTTrainer, SFTConfig




# -------------------------
# Utilitaires données (fallback)
# -------------------------
def pil_loader(path: str) -> Image.Image:
    """Ouvre une image et force le mode RGB pour éviter des surprises (RGBA, L, etc.)."""
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
    Rend le script independant du cwd de lancement.
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
    """
    Verifie que toutes les images existent avant l'entrainement (fail fast).
    """
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
    Important avec datasets: les structs peuvent injecter des cles None
    (ex: {'type':'text','image':None,'text':'...'}), ce qui peut creer
    des placeholders image fantomes.
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




# -------------------------
# Vérifications (sanity checks)
# -------------------------
def schema_sanity_check(ex: Dict[str, Any]) -> None:
    """Vérification minimale du schéma : présence de 'images' et 'messages', et formats de base."""
    if "images" not in ex or "messages" not in ex:
        raise ValueError("Chaque ligne JSONL doit contenir les clés : 'images' et 'messages'.")


    if not isinstance(ex["images"], list) or len(ex["images"]) != 1:
        raise ValueError("'images' doit être une liste de longueur 1 (une image par exemple).")


    if not isinstance(ex["messages"], list) or len(ex["messages"]) < 2:
        raise ValueError("'messages' doit être une liste avec au moins 2 tours (user + assistant).")




def processor_sanity_check(processor: AutoProcessor, ex: Dict[str, Any]) -> None:
    """
    Test léger : encode un exemple et vérifie la présence de tenseurs liés à l'image.
    Le but est de détecter tôt un pipeline qui ne consomme pas l'image (text-only involontaire).
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


            # Cas "text-only": content est une chaîne
            if isinstance(c, str):
                texts.append(f"{role}: {c}")
                continue


            # Cas instruct multi-parties: content est une liste de dicts
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(f"{role}: {part['text']}")
                        break


        text = "\n".join(texts) if texts else "user: (vide)\nassistant: (vide)"


        enc = processor(text=text, images=img0, return_tensors="pt")
        keys = set(enc.keys())


        # Heuristique : clés fréquemment observées dans les VLM (peut varier selon versions)
        vision_like = {"pixel_values", "image_grid_thw", "vision_pixel_values"}
        has_vision = any(k in keys for k in vision_like) or any(("pixel" in k) or ("image" in k) for k in keys)


        if not has_vision:
            print(
                "[ATTENTION] Encodage processor : aucune clé 'vision' évidente détectée.\n"
                "            Cela peut indiquer que l'image n'est pas consommée correctement.\n"
                f"            Clés retournées: {sorted(list(keys))}\n"
                "            Conseil: verifiez les versions Transformers/TRL et le format messages/images."
            )
        else:
            print(f"[OK] Sanity check processor : clés détectées (aperçu) = {sorted(list(keys))[:12]} ...")


    except Exception as e:
        print(
            "[ATTENTION] Sanity check processor a échoué (exception).\n"
            "            Ce n'est pas toujours bloquant, mais c'est un signal de compatibilité.\n"
            f"            Erreur: {repr(e)}\n"
            "            Conseil: verifiez les versions Transformers/TRL et le format messages/images."
        )




# -------------------------
# Arguments CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()


    # Données
    p.add_argument("--train_jsonl", type=str, default="train.jsonl", help="Chemin vers train.jsonl")
    p.add_argument("--val_jsonl", type=str, default="val.jsonl", help="Chemin vers val.jsonl (optionnel)")
    p.add_argument(
        "--use_val_if_exists",
        type=str2bool,
        default=True,
        help="Utiliser val.jsonl si present (true/false).",
    )


    # Modèle
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")


    # Sortie / reproductibilité
    p.add_argument("--output_dir", type=str, default="qwen3_vl_timel_sft_out")
    p.add_argument("--seed", type=int, default=42)


    # Entraînement (par pas)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--per_device_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.05)


    # Contrôle de la troncature (VLM)
    p.add_argument(
        "--max_length",
        type=str,
        default="none",
        help="Mettre 'none' pour max_length=None (recommandé VLM), sinon un entier.",
    )


    # Logs / sauvegarde / éval
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)


    # Precision
    p.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Precision d'entrainement.",
    )


    # Reprise
    p.add_argument("--resume", type=str2bool, default=True, help="Reprendre depuis checkpoint si present.")


    # Divers
    p.add_argument("--report_to", type=str, default="none")  # "wandb" si besoin
    p.add_argument("--dataloader_num_workers", type=int, default=0)


    # Device
    p.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map pour from_pretrained (ex: auto, cuda:0).",
    )


    return p.parse_args()




# -------------------------
# Main
# -------------------------

class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = []
        for f in features:
            # Priorite au schema natif (messages) pour eviter toute desynchronisation
            # potentielle de la colonne text dans certains chemins TRL.
            if isinstance(f.get("messages"), list):
                msg = sanitize_messages_for_template(f["messages"])
                t = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
            else:
                t = f["text"]
            texts.append(t)

        imgs = []
        for f in features:
            im = f["images"]
            if isinstance(im, list):
                im = im[0]
            # Evite les ambiguities de parsing (path/url/base64) cote processor:
            # on charge explicitement en PIL pour chaque image.
            if isinstance(im, str):
                im = pil_loader(im)
            imgs.append(im)

        # Garde-fou: dans ce pipeline, chaque exemple doit avoir exactement 1 image
        # donc 1 token <|vision_start|> dans le texte applique via chat_template.
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

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)


    os.makedirs(args.output_dir, exist_ok=True)


    # Chargement des datasets JSONL
    if not os.path.exists(args.train_jsonl):
        raise FileNotFoundError(f"train_jsonl introuvable: {args.train_jsonl}")


    data_files = {"train": args.train_jsonl}
    use_val = args.use_val_if_exists and os.path.exists(args.val_jsonl)
    if use_val:
        data_files["validation"] = args.val_jsonl


    ds = load_dataset("json", data_files=data_files)


    # Vérification schéma minimale (sur le 1er exemple)
    first = ds["train"][0]
    schema_sanity_check(first)

    # Rend les chemins independants du dossier courant:
    # - train split resolu depuis le dossier de train_jsonl
    # - val split resolu depuis le dossier de val_jsonl
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

    # Recharge le premier exemple apres resolution des chemins
    first = ds["train"][0]


    # Chargement processor & modèle (style script 2)
    processor = AutoProcessor.from_pretrained(args.model_name)

    def build_text(ex):
        # Daniel van Strien style: chat template inserts correct image placeholder tokens
        msg = sanitize_messages_for_template(ex["messages"])
        txt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        return {"text": txt}

    ds["train"] = ds["train"].map(build_text, desc="chat_template(train)")
    if use_val:
        ds["validation"] = ds["validation"].map(build_text, desc="chat_template(val)")

    torch_dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype_map[args.precision],
        device_map=args.device_map,
    )


    # Sanity check processor (avertissement uniquement)
    ex_for_check = first
    processor_sanity_check(processor, ex_for_check)


    # max_length: "none" -> None
    max_length: Optional[int]
    if args.max_length.lower() == "none":
        max_length = None
    else:
        max_length = int(args.max_length)


    # Configuration TRL
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
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        remove_unused_columns=False,  # important VLM
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        max_length=max_length,  # None recommandé en VLM
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


    # Reprise si checkpoints détectés
    resume_from: Optional[str] = None
    if args.resume and os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            resume_from = args.output_dir


    trainer.train(resume_from_checkpoint=resume_from)


    # Sauvegarde finale
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)


    print(f"[TERMINE] Modèle final sauvegardé dans: {final_dir}")




if __name__ == "__main__":
    main()
