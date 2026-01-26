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
- Option --normalize_pil : mode de repli si votre stack (TRL/Transformers/processor)
  ne supporte pas correctement les messages multi-parties (image+texte) dans `messages.content`.
  Dans ce mode, on convertit :
    - images[0] (chemin) -> PIL.Image
    - messages -> conversation textuelle (content: str)
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




def normalize_example_to_pil_textonly(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mode de repli : convertit un exemple JSONL "instruct" vers une forme
    très robuste pour le collator VLM de TRL.


    Attendu en entrée (schéma courant) :
      - ex["images"] == ["/chemin/vers/image.jpg"]
      - ex["messages"] est une liste de tours, où content est souvent une liste
        de parties: [{"type":"image"}, {"type":"text","text":"..."}]


    Sortie :
      - images: [PIL.Image]
      - messages: [{"role":"user","content": str}, {"role":"assistant","content": str}]
    """
    img_path = ex["images"][0]
    img = pil_loader(img_path)


    user_text: Optional[str] = None
    assistant_text: Optional[str] = None


    for m in ex.get("messages", []):
        role = m.get("role")
        content = m.get("content", [])


        if role == "user":
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        user_text = c.get("text")
                        break
            elif isinstance(content, str):
                user_text = content


        elif role == "assistant":
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        assistant_text = c.get("text")
                        break
            elif isinstance(content, str):
                assistant_text = content


    if user_text is None or assistant_text is None:
        raise ValueError("Exemple invalide : impossible d'extraire user_text/assistant_text depuis messages.")


    return {
        "images": [img],
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }




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
                "            Conseil: réessayez avec --normalize_pil si vous observez un entraînement text-only."
            )
        else:
            print(f"[OK] Sanity check processor : clés détectées (aperçu) = {sorted(list(keys))[:12]} ...")


    except Exception as e:
        print(
            "[ATTENTION] Sanity check processor a échoué (exception).\n"
            "            Ce n'est pas toujours bloquant, mais c'est un signal de compatibilité.\n"
            f"            Erreur: {repr(e)}\n"
            "            Conseil: relancez avec --normalize_pil si vous avez des erreurs de collator/schema."
        )




# -------------------------
# Arguments CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()


    # Données
    p.add_argument("--train_jsonl", type=str, default="train.jsonl", help="Chemin vers train.jsonl")
    p.add_argument("--val_jsonl", type=str, default="val.jsonl", help="Chemin vers val.jsonl (optionnel)")
    p.add_argument("--use_val_if_exists", action="store_true", default=True, help="Utiliser val.jsonl si présent")


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


    # Précision
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)


    # Reprise
    p.add_argument("--resume", action="store_true", default=True)


    # Divers
    p.add_argument("--report_to", type=str, default="none")  # "wandb" si besoin
    p.add_argument("--dataloader_num_workers", type=int, default=0)


    # Mode de repli compatibilité
    p.add_argument(
        "--normalize_pil",
        action="store_true",
        default=False,
        help="Repli: image path->PIL + messages text-only (plus robuste, potentiellement plus de RAM).",
    )


    return p.parse_args()




# -------------------------
# Main
# -------------------------

class VLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = [f["text"] for f in features]

        imgs = []
        for f in features:
            im = f["images"]
            if isinstance(im, list):
                im = im[0]
            imgs.append(im)

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


    # Mode repli: normaliser vers PIL + text-only
    if args.normalize_pil:
        train_cols = ds["train"].column_names
        ds["train"] = ds["train"].map(
            normalize_example_to_pil_textonly,
            remove_columns=train_cols,
            desc="Normalisation: image path->PIL, messages->texte pur",
        )
        if use_val:
            val_cols = ds["validation"].column_names
            ds["validation"] = ds["validation"].map(
                normalize_example_to_pil_textonly,
                remove_columns=val_cols,
                desc="Normalisation (val): image path->PIL, messages->texte pur",
            )


        schema_sanity_check(ds["train"][0])


    # Chargement processor & modèle (style script 2)
    processor = AutoProcessor.from_pretrained(args.model_name)

    def build_text(ex):
        # Daniel van Strien style: chat template inserts correct image placeholder tokens
        txt = processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": txt}

    # Only for the normal path (messages are multipart). Do NOT use with --normalize_pil
    if not args.normalize_pil:
        ds["train"] = ds["train"].map(build_text, desc="chat_template(train)")
        if use_val:
            ds["validation"] = ds["validation"].map(build_text, desc="chat_template(val)")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="cuda:0",
    )


    # Sanity check processor (avertissement uniquement)
    ex_for_check = ds["train"][0] if args.normalize_pil else first
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
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        remove_unused_columns=False,  # important VLM
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        max_length=max_length,  # None recommandé en VLM
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