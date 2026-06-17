#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen3vl_sft_timel_merged_fr_V6.py — ESQUISSE (path-based supervision)

Script de fine-tuning SFT pour Qwen3-VL (étiquettes timel), version V6.

Changements V6 (vs V5) :
- **Path-based supervision** : à l'entraînement, chaque label est sérialisé sous forme
  de chemin complet depuis sa racine taxonomique, avec ' / ' comme séparateur interne
  et ' ; ' comme séparateur entre labels.
  Exemple V5 :  "ablette, château"
  Exemple V6 :  "nature lieu / animaux / animaux aquatiques / ablette ; objet architecture / construction / château"
  → noeuds intermédiaires fréquents partagés entre tous les leaves d'une même branche
  → signal d'apprentissage stable même pour les feuilles à 1–5 exemples.
- **Trie path-based** au décodage contraint : indexé sur les chemins complets, séparateur
  ' ; ' entre paths, ' / ' à l'intérieur d'un path. Préfixes partagés au niveau du token.
- **Parser inverse** à l'inférence : la sortie path-formattée est re-projetée vers les
  IDs feuille (taxonomie chargée depuis timel_taxonomy.json).
- **Métriques hiérarchiques** intégrées : micro H-P/R/F1 sur l'union des ancêtres.

V5 conservé intact comme baseline.

Statut : ESQUISSE — testée pour la cohérence syntaxique et la logique de sérialisation,
mais nécessite un round d'entraînement pour valider en bout-en-bout.

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
import unicodedata
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from datasets import load_dataset

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    LogitsProcessor,
    LogitsProcessorList,
    TrainerCallback,
    set_seed,
)
from trl import SFTTrainer, SFTConfig

from metrics_soft import (
    compute_ic,
    ic_weighted_soft_prf,
    label_counts_from_jsonl,
    soft_prf,
)


DEFAULT_SYSTEM_PROMPT = (
    "Tu es un classifieur visuel iconographique. Liste les étiquettes TIMEL "
    "pertinentes pour cette image SOUS FORME DE CHEMIN HIÉRARCHIQUE COMPLET, "
    "depuis la racine taxonomique jusqu'à la feuille. Utilise ' / ' entre les "
    "niveaux et ' ; ' entre étiquettes distinctes. N'écris aucun commentaire, "
    "aucun doublon, et utilise exactement les libellés canoniques. Termine ta "
    "sortie après la dernière feuille."
)

# Séparateurs path-based (V6)
PATH_SEP = " / "      # entre niveaux taxonomiques d'un même chemin
LABELS_SEP = " ; "    # entre chemins distincts


TERM_SPLIT_RE = re.compile(r"[,;\n\r]+")


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


def inject_system_prompt(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    """Prepend un tour 'system' si demandé et qu'il n'y en a pas déjà un."""
    if not system_prompt:
        return messages
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        return messages
    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}],
    }
    return [system_msg] + list(messages)


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


def normalize_term_key(text: str) -> str:
    """Normalise une étiquette pour le matching vocabulaire contrôlé."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.replace("’", "'").replace("`", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\"'“”‘’.,;:!?-]+", "", text)
    text = re.sub(r"[\s\"'“”‘’.,;:!?-]+$", "", text)
    return text


def load_timel_term_reference(
    classes_tsv: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Charge le référentiel des labels TIMEL depuis classes.tsv.

    Retourne (normalized_to_canonical, canonical_to_id, id_to_label).
    """
    if not classes_tsv:
        raise ValueError("classes_tsv requis pour utiliser le referentiel terms.")
    if not os.path.isfile(classes_tsv):
        raise FileNotFoundError(f"classes_tsv introuvable: {classes_tsv}")

    normalized_to_canonical: Dict[str, str] = {}
    canonical_to_id: Dict[str, str] = {}
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
            if tid and label:
                key = normalize_term_key(label)
                normalized_to_canonical[key] = label
                canonical_to_id[label] = tid
                id_to_label[tid] = label

    if not normalized_to_canonical:
        raise ValueError(f"Aucun timel_label charge depuis {classes_tsv}")
    return normalized_to_canonical, canonical_to_id, id_to_label


def load_timel_taxonomy(
    taxonomy_json: str,
    id_to_label: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Charge la taxonomie hiérarchique TIMEL.

    Retourne :
        - id_to_path_ids   : leaf_id → liste d'IDs de la racine à la feuille (incluse)
        - id_to_path_labels: leaf_id → liste de labels canoniques correspondants
    Quand un ID apparaît dans plusieurs entrées (DAG), on conserve la première occurrence.
    Les nœuds dont le label canonique est inconnu (absent de classes.tsv) sont quand
    même conservés dans path_labels — leur label est pris depuis le champ 'value' de
    l'entrée taxonomie.
    """
    if not taxonomy_json or not os.path.isfile(taxonomy_json):
        raise FileNotFoundError(f"taxonomy_json introuvable: {taxonomy_json}")

    with open(taxonomy_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Map id → label (fallback sur 'value' si l'id n'est pas une feuille classes.tsv)
    id_to_value: Dict[str, str] = {}
    for it in data.get("items", []):
        path = it.get("path_ids") or []
        if not path:
            continue
        leaf = path[-1]
        if leaf not in id_to_value:
            id_to_value[leaf] = (it.get("value") or "").strip()

    id_to_path_ids: Dict[str, List[str]] = {}
    id_to_path_labels: Dict[str, List[str]] = {}
    for it in data.get("items", []):
        path = it.get("path_ids") or []
        if not path:
            continue
        leaf = path[-1]
        if leaf in id_to_path_ids:
            continue
        id_to_path_ids[leaf] = list(path)
        labels: List[str] = []
        for pid in path:
            lab = id_to_label.get(pid) or id_to_value.get(pid) or pid
            labels.append(lab)
        id_to_path_labels[leaf] = labels

    return id_to_path_ids, id_to_path_labels


def serialize_label_to_path(
    leaf_label: str,
    canonical_to_id: Dict[str, str],
    id_to_path_labels: Dict[str, List[str]],
    path_sep: str = PATH_SEP,
) -> str:
    """
    Sérialise un leaf label vers sa représentation path-based.

    Si le label n'est pas dans canonical_to_id ou la taxonomie, on retourne le
    label brut (fallback safe).
    """
    leaf_id = canonical_to_id.get(leaf_label)
    if leaf_id is None:
        return leaf_label
    path = id_to_path_labels.get(leaf_id)
    if not path:
        return leaf_label
    return path_sep.join(path)


def serialize_labels_to_paths(
    leaf_labels: List[str],
    canonical_to_id: Dict[str, str],
    id_to_path_labels: Dict[str, List[str]],
    path_sep: str = PATH_SEP,
    labels_sep: str = LABELS_SEP,
) -> str:
    """Sérialise une liste de leaf labels vers le texte assistant path-based."""
    chunks = [
        serialize_label_to_path(lbl, canonical_to_id, id_to_path_labels, path_sep)
        for lbl in leaf_labels
    ]
    return labels_sep.join(chunks)


def parse_path_output(
    text: str,
    normalized_to_canonical: Dict[str, str],
    path_sep: str = PATH_SEP,
    labels_sep: str = LABELS_SEP,
) -> Dict[str, List[str]]:
    """
    Parse une sortie générée au format path-based.

    Stratégie :
        1. Split sur labels_sep → chunks (= chemins)
        2. Pour chaque chunk, split sur path_sep → segments
        3. Le dernier segment = leaf candidat
        4. Match contre normalized_to_canonical (insensible casse/accents/ponctuation)
        5. Conserve l'ordre d'apparition, élimine les doublons
    """
    if not isinstance(text, str) or not text.strip():
        return {"extracted_terms": [], "valid_terms": [], "invalid_terms": []}

    # Strip de bord et nettoyage : tolérant aux séparateurs alternatifs.
    raw_chunks: List[str] = []
    for chunk in text.split(labels_sep):
        chunk = chunk.strip()
        if chunk:
            raw_chunks.append(chunk)

    seen: Set[str] = set()
    extracted: List[str] = []
    valid: List[str] = []
    invalid: List[str] = []

    for chunk in raw_chunks:
        # Le leaf = dernier segment après path_sep
        segments = [seg.strip() for seg in chunk.split(path_sep) if seg.strip()]
        if not segments:
            continue
        leaf_candidate = segments[-1]
        extracted.append(leaf_candidate)
        key = normalize_term_key(leaf_candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in normalized_to_canonical:
            valid.append(normalized_to_canonical[key])
        else:
            invalid.append(leaf_candidate)

    return {
        "extracted_terms": extracted,
        "valid_terms": valid,
        "invalid_terms": invalid,
    }


def rewrite_messages_to_paths(
    messages: List[Dict[str, Any]],
    canonical_to_id: Dict[str, str],
    id_to_path_labels: Dict[str, List[str]],
    normalized_to_canonical: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Pour chaque message assistant, transforme le contenu texte (liste de leaf
    labels séparés par virgule) en texte path-based.

    Tolérant : labels absents de la taxonomie restent en l'état (fallback).
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, list):
            new_parts: List[Dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    raw_terms = split_term_candidates(part["text"])
                    # Normalise vers canonical avant sérialisation.
                    canon_leaves: List[str] = []
                    seen_local: Set[str] = set()
                    for t in raw_terms:
                        key = normalize_term_key(t)
                        if not key or key in seen_local:
                            continue
                        seen_local.add(key)
                        canon = normalized_to_canonical.get(key, t.strip())
                        canon_leaves.append(canon)
                    new_text = serialize_labels_to_paths(
                        canon_leaves, canonical_to_id, id_to_path_labels
                    )
                    np = dict(part)
                    np["text"] = new_text
                    new_parts.append(np)
                else:
                    new_parts.append(part)
            m["content"] = new_parts
        elif isinstance(content, str):
            raw_terms = split_term_candidates(content)
            canon_leaves: List[str] = []
            seen_local: Set[str] = set()
            for t in raw_terms:
                key = normalize_term_key(t)
                if not key or key in seen_local:
                    continue
                seen_local.add(key)
                canon = normalized_to_canonical.get(key, t.strip())
                canon_leaves.append(canon)
            m["content"] = serialize_labels_to_paths(
                canon_leaves, canonical_to_id, id_to_path_labels
            )
        out.append(m)
    return out


def split_term_candidates(text: str) -> List[str]:
    """Découpe une prédiction libre en candidats termes."""
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    for piece in TERM_SPLIT_RE.split(text):
        piece = piece.strip()
        if piece:
            chunks.append(piece)
    return chunks


def normalize_prediction_terms(
    text: str,
    normalized_to_canonical: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    Normalise la sortie générée :
    - découpe en termes candidats
    - compare après normalisation textuelle
    - conserve l'ordre d'apparition
    - enlève les doublons
    - sépare termes valides / invalides selon le référentiel
    """
    seen: Set[str] = set()
    valid: List[str] = []
    invalid: List[str] = []
    extracted = split_term_candidates(text)

    for term in extracted:
        key = normalize_term_key(term)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)

        if key in normalized_to_canonical:
            valid.append(normalized_to_canonical[key])
        else:
            invalid.append(term)

    return {
        "extracted_terms": extracted,
        "valid_terms": valid,
        "invalid_terms": invalid,
    }


def extract_gold_terms_from_example(
    ex: Dict[str, Any],
    normalized_to_canonical: Dict[str, str],
) -> List[str]:
    """Récupère les termes gold déjà présents dans un exemple annoté."""
    for msg in ex.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return normalize_prediction_terms(
                    part.get("text", ""),
                    normalized_to_canonical,
                )["valid_terms"]
    return []


# -------------------------
# Décodage contraint par trie (V5)
# -------------------------

class LabelTrie:
    """
    Trie sur les séquences de token-ids des labels canoniques.

    Chaque label est inséré dans deux variantes pour absorber l'influence du
    leading-space des tokenizers BPE :
        - variante "fresh"     : tokenize(label)
        - variante "après sep" : tokenize(", " + label)[len(sep_tokens):]
                                 (i.e. ce qui suit la séquence de séparateur)

    Les deux variantes peuvent partager des préfixes ou non. On accepte
    n'importe laquelle au démarrage d'un label.

    Structure d'un noeud : {"children": {token_id: node}, "terminal": bool, "label": Optional[str]}.
    Un noeud terminal porte le label canonique complet pour pouvoir reporter
    quel label vient d'être émis (utile pour interdire les doublons).
    """

    def __init__(self, labels: List[str], tokenizer, separator: str = ", "):
        self.tokenizer = tokenizer
        self.separator = separator
        self.root: Dict[str, Any] = {"children": {}, "terminal": False, "label": None}
        self.sep_tokens: List[int] = tokenizer.encode(separator, add_special_tokens=False)
        if not self.sep_tokens:
            raise ValueError(f"Le séparateur {separator!r} s'est tokenisé en séquence vide.")

        self.n_nodes = 1
        self.max_depth = 0

        seen_paths: Set[Tuple[int, ...]] = set()
        for label in labels:
            variants: List[List[int]] = []
            v1 = tokenizer.encode(label, add_special_tokens=False)
            if v1:
                variants.append(v1)
            with_sep = tokenizer.encode(separator + label, add_special_tokens=False)
            if with_sep[: len(self.sep_tokens)] == self.sep_tokens:
                v2 = with_sep[len(self.sep_tokens):]
                if v2 and v2 != v1:
                    variants.append(v2)
            for path in variants:
                path_t = tuple(path)
                if path_t in seen_paths:
                    continue
                seen_paths.add(path_t)
                self._insert(path, label)

    def _insert(self, path: List[int], label: str) -> None:
        node = self.root
        for tok in path:
            if tok not in node["children"]:
                node["children"][tok] = {"children": {}, "terminal": False, "label": None}
                self.n_nodes += 1
            node = node["children"][tok]
        node["terminal"] = True
        node["label"] = label
        if len(path) > self.max_depth:
            self.max_depth = len(path)

    def transitions(self, node: Dict[str, Any]) -> Set[int]:
        return set(node["children"].keys())


class LabelTrieLogitsProcessor(LogitsProcessor):
    """
    Force la génération à parcourir le trie des labels canoniques.

    Par séquence du batch on maintient un état :
        - current_node : position dans le trie (None = root, en cours d'émission d'un label)
        - in_sep_idx   : si > 0, on est en train d'émettre la séquence séparateur
        - emitted_labels : set des labels déjà complétés (pour interdire les doublons)
        - done : True après EOS / im_end (plus aucune contrainte)

    Cet objet est mono-shot : à utiliser pour UN appel à generate() avec un prompt
    de longueur connue (prompt_len). On reconstitue l'état à chaque step en
    rejouant les tokens générés (input_ids[:, prompt_len:]).
    """

    def __init__(
        self,
        trie: LabelTrie,
        tokenizer,
        prompt_len: int,
        forbid_repeat: bool = True,
    ):
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.forbid_repeat = forbid_repeat

        self.eos_token_id: Optional[int] = tokenizer.eos_token_id
        # Qwen3 utilise <|im_end|> pour clôturer un tour de chat.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id == tokenizer.unk_token_id:
            im_end_id = None
        self.im_end_token_id: Optional[int] = im_end_id
        self.stop_ids: Set[int] = {
            tid for tid in (self.eos_token_id, self.im_end_token_id) if tid is not None
        }
        if not self.stop_ids:
            raise ValueError("Aucun token d'arrêt (eos / im_end) trouvé.")

    def _replay_state(self, gen_ids: List[int]) -> Tuple[Optional[Dict[str, Any]], int, Set[str], bool]:
        """
        Rejoue les tokens générés pour reconstituer l'état du trie.

        Retourne (current_node, in_sep_idx, emitted_labels, done).
            - current_node : None si on est à la racine (prêt à démarrer un label) ;
                             sinon noeud du trie en cours.
            - in_sep_idx   : index dans self.trie.sep_tokens si on est dans la séquence sep.
                             0 sinon.
            - emitted_labels : set des labels complétés vus jusqu'ici.
            - done : True si un token d'arrêt a été émis.
        """
        node: Optional[Dict[str, Any]] = None  # None = at root, awaiting a label start
        in_sep_idx = 0
        emitted: Set[str] = set()
        done = False
        sep_tokens = self.trie.sep_tokens

        for tok in gen_ids:
            if done:
                # Une fois EOS émis on ne contraint plus.
                break
            if tok in self.stop_ids:
                done = True
                continue

            if in_sep_idx > 0:
                # On était en plein milieu de la séquence séparateur.
                if tok != sep_tokens[in_sep_idx]:
                    # Ne devrait jamais arriver puisque l'on masque, mais on reste safe.
                    in_sep_idx = 0
                    node = None
                    continue
                in_sep_idx += 1
                if in_sep_idx == len(sep_tokens):
                    in_sep_idx = 0
                    node = None
                continue

            if node is None:
                # On démarre un label : tok doit être un enfant de root.
                child = self.trie.root["children"].get(tok)
                if child is None:
                    # Token inattendu (masqué normalement). On force un reset propre.
                    node = None
                    continue
                node = child
                if node["terminal"] and not node["children"]:
                    # Label feuille pure : on l'enregistre et on revient à la racine
                    # implicitement en attendant un séparateur ou un stop.
                    if node["label"] is not None:
                        emitted.add(node["label"])
                continue

            # On est mid-trie. tok peut être :
            #   - un enfant (continue dans le trie)
            #   - le premier token du séparateur si le noeud courant est terminal
            if tok in node["children"]:
                node = node["children"][tok]
                if node["terminal"] and not node["children"]:
                    if node["label"] is not None:
                        emitted.add(node["label"])
                continue

            if node["terminal"] and tok == sep_tokens[0]:
                # Fin de label, début de séparateur.
                if node["label"] is not None:
                    emitted.add(node["label"])
                if len(sep_tokens) == 1:
                    node = None
                    in_sep_idx = 0
                else:
                    in_sep_idx = 1
                    node = None  # on est officiellement entre deux labels
                continue

            # Token inattendu → reset défensif.
            node = None
            in_sep_idx = 0

        return node, in_sep_idx, emitted, done

    def _allowed_tokens(
        self,
        node: Optional[Dict[str, Any]],
        in_sep_idx: int,
        emitted: Set[str],
    ) -> Set[int]:
        sep_tokens = self.trie.sep_tokens

        if in_sep_idx > 0:
            # On doit continuer la séquence séparateur.
            return {sep_tokens[in_sep_idx]}

        if node is None:
            # On est à la racine, en attente d'un label.
            # Autorisé : (a) tout premier-token de label NON ENCORE EMIS, (b) stop.
            allowed: Set[int] = set()
            for tok, child in self.trie.root["children"].items():
                if not self.forbid_repeat:
                    allowed.add(tok)
                    continue
                # Si forbid_repeat : on regarde si au moins un label sous ce préfixe
                # n'a pas encore été émis. Approximation rapide : on autorise toujours,
                # et on filtrera plus profondément. Le doublonnage exact est appliqué
                # via no_repeat_ngram_size côté HF + le filtre terminal ci-dessous.
                allowed.add(tok)
            # Stop autorisé (le modèle peut décider de s'arrêter).
            allowed |= self.stop_ids
            return allowed

        # Mid-trie : enfants du noeud courant + éventuellement séparateur si terminal.
        allowed = set(node["children"].keys())
        if node["terminal"]:
            allowed.add(sep_tokens[0])
            # Le modèle peut aussi clore directement après ce label.
            allowed |= self.stop_ids
            # Si on interdit les doublons : exclure les enfants menant uniquement
            # à des labels déjà émis (filtre conservateur : on ne masque pas pour rester
            # tractable ; no_repeat_ngram_size complète la protection).
        return allowed

    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor") -> "torch.FloatTensor":
        batch_size, vocab_size = scores.shape
        device = scores.device

        mask = torch.full_like(scores, float("-inf"))

        for i in range(batch_size):
            gen_ids = input_ids[i, self.prompt_len:].tolist()
            node, in_sep_idx, emitted, done = self._replay_state(gen_ids)
            if done:
                # Plus aucune contrainte : on laisse passer tous les tokens.
                mask[i, :] = 0.0
                continue
            allowed = self._allowed_tokens(node, in_sep_idx, emitted)
            if not allowed:
                # Cas pathologique : on autorise au moins l'EOS pour ne pas bloquer.
                allowed = set(self.stop_ids)
            idx = torch.tensor(sorted(allowed), dtype=torch.long, device=device)
            mask[i, idx] = 0.0

        return scores + mask


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

    # Inférence (V6 : path-based, sortie ~4-6× plus longue qu'en V5 flat)
    p.add_argument("--max_new_tokens", type=int, default=1280)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="Pénalité de répétition (>1 décourage les répétitions).",
    )
    p.add_argument(
        "--no_repeat_ngram_size", type=int, default=5,
        help="Interdit la répétition exacte de n-grams de cette taille.",
    )
    p.add_argument(
        "--constrained_decoding", type=str2bool, default=True,
        help="Active le décodage contraint par trie sur les labels canoniques.",
    )

    # System prompt (V5)
    p.add_argument(
        "--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
        help="Prompt système injecté en train ET en inférence si absent du JSONL. "
             "Vide ('') pour désactiver l'injection.",
    )

    # Path-based supervision (V6)
    p.add_argument(
        "--taxonomy_json", type=str, default="",
        help="Chemin vers timel_taxonomy.json. Requis pour --path_mode true.",
    )
    p.add_argument(
        "--path_mode", type=str2bool, default=True,
        help="Active la supervision path-based : assistant content réécrit en "
             "chemins hiérarchiques complets, et trie indexé sur ces chemins.",
    )

    # Callback eval soft hiérarchique (V6)
    p.add_argument(
        "--soft_eval_samples", type=int, default=0,
        help="Nombre d'exemples du val sur lesquels lancer une éval générative "
             "soft pendant l'entraînement (0 = désactivé).",
    )
    p.add_argument(
        "--soft_eval_ic", type=str2bool, default=True,
        help="Active la pondération par IC dans l'éval soft pendant training.",
    )
    p.add_argument(
        "--ic_alpha", type=float, default=1.0,
        help="Lissage additif (Laplace) pour le calcul des IC.",
    )

    return p.parse_args()


# -------------------------
# Data Collator
# -------------------------

class VLMDataCollator:
    def __init__(
        self,
        processor,
        system_prompt: Optional[str] = None,
        canonical_to_id: Optional[Dict[str, str]] = None,
        id_to_path_labels: Optional[Dict[str, List[str]]] = None,
        normalized_to_canonical: Optional[Dict[str, str]] = None,
        path_mode: bool = False,
    ):
        self.processor = processor
        self.system_prompt = system_prompt or None
        self.path_mode = bool(path_mode)
        self.canonical_to_id = canonical_to_id or {}
        self.id_to_path_labels = id_to_path_labels or {}
        self.normalized_to_canonical = normalized_to_canonical or {}

    def __call__(self, features):
        # FIX: apply_chat_template appelé une seule fois ici (build_text supprimé)
        texts = []
        for f in features:
            if isinstance(f.get("messages"), list):
                msg = inject_system_prompt(f["messages"], self.system_prompt)
                if self.path_mode:
                    msg = rewrite_messages_to_paths(
                        msg,
                        self.canonical_to_id,
                        self.id_to_path_labels,
                        self.normalized_to_canonical,
                    )
                msg = sanitize_messages_for_template(msg)
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

# -------------------------
# Callback éval soft hiérarchique (V6)
# -------------------------

class SoftHierarchicalEvalCallback(TrainerCallback):
    """
    À chaque évaluation, génère sur un sous-ensemble val (taille fixée),
    parse la sortie path-based, et logge soft_f1 + IC-weighted soft_f1.

    Le coût est proportionnel à `max_samples` × max_new_tokens. Garder
    petit (50–200) pour ne pas exploser l'overhead.
    """

    def __init__(
        self,
        trainer_ref,
        processor,
        eval_dataset,
        system_prompt: Optional[str],
        trie: Optional[LabelTrie],
        canonical_to_id: Dict[str, str],
        normalized_to_canonical: Dict[str, str],
        id_to_path_ids: Dict[str, List[str]],
        ic_table: Optional[Dict[str, float]],
        max_new_tokens: int,
        max_samples: int,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 5,
        num_beams: int = 1,
    ):
        self.trainer_ref = trainer_ref
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.system_prompt = system_prompt or None
        self.trie = trie
        self.canonical_to_id = canonical_to_id
        self.normalized_to_canonical = normalized_to_canonical
        self.id_to_path_ids = id_to_path_ids
        self.ic_table = ic_table or {}
        self.max_new_tokens = int(max_new_tokens)
        self.max_samples = int(max_samples)
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_beams = num_beams

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            model = getattr(self.trainer_ref, "model", None)
        if model is None or self.eval_dataset is None or self.max_samples <= 0:
            return
        n = min(self.max_samples, len(self.eval_dataset))
        if n == 0:
            return

        model.eval()
        soft_p_sum = soft_r_sum = soft_f1_sum = 0.0
        ic_p_sum = ic_r_sum = ic_f1_sum = 0.0
        n_done = 0

        for i in range(n):
            ex = self.eval_dataset[i]
            try:
                prompt_messages = inject_system_prompt(
                    keep_prompt_messages_only(ex["messages"]),
                    self.system_prompt,
                )
                prompt_messages = sanitize_messages_for_template(prompt_messages)
                text = self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                )
                img = pil_loader(ex["images"][0])
                batch = self.processor(
                    text=[text], images=[img], return_tensors="pt", padding=True,
                )
                batch = move_batch_to_model_device(batch, model)
                prompt_len = batch["input_ids"].shape[1]
                logits_processors = None
                if self.trie is not None:
                    logits_processors = LogitsProcessorList([
                        LabelTrieLogitsProcessor(
                            trie=self.trie,
                            tokenizer=self.processor.tokenizer,
                            prompt_len=prompt_len,
                        )
                    ])
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=self.num_beams,
                        repetition_penalty=self.repetition_penalty,
                        no_repeat_ngram_size=self.no_repeat_ngram_size,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        logits_processor=logits_processors,
                    )
                generated_new = generated[:, prompt_len:]
                raw_text = self.processor.batch_decode(
                    generated_new, skip_special_tokens=True,
                )[0].strip()
                parsed = parse_path_output(raw_text, self.normalized_to_canonical)
                pred_terms = parsed["valid_terms"]
                gold_terms = extract_gold_terms_from_example(ex, self.normalized_to_canonical)

                pred_ids = [self.canonical_to_id[t] for t in pred_terms if t in self.canonical_to_id]
                gold_ids = [self.canonical_to_id[t] for t in gold_terms if t in self.canonical_to_id]
                if not pred_ids and not gold_ids:
                    continue
                sp, sr, sf = soft_prf(pred_ids, gold_ids, self.id_to_path_ids)
                soft_p_sum += sp
                soft_r_sum += sr
                soft_f1_sum += sf
                if self.ic_table:
                    ip, ir, ifr = ic_weighted_soft_prf(
                        pred_ids, gold_ids, self.id_to_path_ids, self.ic_table,
                    )
                    ic_p_sum += ip
                    ic_r_sum += ir
                    ic_f1_sum += ifr
                n_done += 1
            except Exception as e:
                print(f"[CALLBACK WARN] example {i} skipped: {repr(e)}")

        if n_done == 0:
            return

        logs = {
            "eval_soft_p": soft_p_sum / n_done,
            "eval_soft_r": soft_r_sum / n_done,
            "eval_soft_f1": soft_f1_sum / n_done,
            "eval_soft_n": n_done,
        }
        if self.ic_table:
            logs.update({
                "eval_ic_soft_p": ic_p_sum / n_done,
                "eval_ic_soft_r": ic_r_sum / n_done,
                "eval_ic_soft_f1": ic_f1_sum / n_done,
            })
        try:
            self.trainer_ref.log(logs)
        except Exception:
            print(f"[CALLBACK] eval soft metrics: {logs}")
        model.train()


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

    # Validation des labels assistant vs classes.tsv (warn si dataset bruyant)
    if args.classes_tsv and os.path.isfile(args.classes_tsv):
        normalized_to_canonical, _ct, _itl = load_timel_term_reference(args.classes_tsv)
        n_examples_scanned = 0
        n_label_valid = 0
        n_label_invalid = 0
        n_examples_with_any_invalid = 0
        max_scan = min(2000, len(ds["train"]))
        for i in range(max_scan):
            ex = ds["train"][i]
            n_examples_scanned += 1
            gold_text = ""
            for msg in ex.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            gold_text = part.get("text", "")
                            break
                break
            extracted = split_term_candidates(gold_text)
            had_invalid = False
            for term in extracted:
                key = normalize_term_key(term)
                if not key:
                    continue
                if key in normalized_to_canonical:
                    n_label_valid += 1
                else:
                    n_label_invalid += 1
                    had_invalid = True
            if had_invalid:
                n_examples_with_any_invalid += 1
        total_lab = n_label_valid + n_label_invalid
        invalid_pct = (n_label_invalid / total_lab * 100.0) if total_lab > 0 else 0.0
        print(
            f"[DATA] Scan {n_examples_scanned} ex. train : "
            f"labels valides={n_label_valid}, invalides={n_label_invalid} "
            f"({invalid_pct:.1f}%), exemples impactés={n_examples_with_any_invalid}"
        )
        if invalid_pct > 5.0:
            print(
                "[DATA WARN] Plus de 5% des labels assistant ne matchent pas classes.tsv. "
                "Vérifiez la normalisation du dataset ou le bon classes_tsv."
            )

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

    # Path-based collator (V6) : nécessite classes.tsv et taxonomy_json si path_mode actif.
    if args.path_mode:
        if not args.classes_tsv or not args.taxonomy_json:
            raise ValueError(
                "--path_mode true requiert --classes_tsv ET --taxonomy_json."
            )
        n2c, c2i, i2l = load_timel_term_reference(args.classes_tsv)
        id_to_path_ids, id_to_path_labels = load_timel_taxonomy(args.taxonomy_json, i2l)
        print(
            f"[TAXO] {len(id_to_path_ids)} chemins indexés "
            f"(profondeur moy={sum(len(v) for v in id_to_path_ids.values()) / max(1, len(id_to_path_ids)):.2f})"
        )
        collator = VLMDataCollator(
            processor,
            system_prompt=args.system_prompt,
            canonical_to_id=c2i,
            id_to_path_labels=id_to_path_labels,
            normalized_to_canonical=n2c,
            path_mode=True,
        )
    else:
        collator = VLMDataCollator(processor, system_prompt=args.system_prompt)

    trainer = SFTTrainer(
        data_collator=collator,
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation") if use_val else None,
        processing_class=processor,
    )

    # Callback éval soft hiérarchique (V6) — requiert use_val + path_mode
    if args.soft_eval_samples > 0 and use_val and args.path_mode:
        # Reload references (idempotent ; mutualise avec le bloc collator path_mode)
        n2c_cb, c2i_cb, i2l_cb = load_timel_term_reference(args.classes_tsv)
        id_to_path_ids_cb, id_to_path_labels_cb = load_timel_taxonomy(
            args.taxonomy_json, i2l_cb
        )
        # IC depuis le train_jsonl (doc-frequency)
        ic_table_cb: Dict[str, float] = {}
        if args.soft_eval_ic:
            counts_cb, N_cb = label_counts_from_jsonl(args.train_jsonl, c2i_cb, n2c_cb)
            ic_table_cb = compute_ic(
                counts=counts_cb,
                total_n=N_cb,
                vocab_size=len(c2i_cb),
                alpha=args.ic_alpha,
            )
            print(f"[CALLBACK/IC] {len(ic_table_cb)} IC calculés (N={N_cb}, α={args.ic_alpha})")
        # Trie path-based pour décodage contraint
        cb_trie: Optional[LabelTrie] = None
        if args.constrained_decoding:
            path_strings_cb = [
                PATH_SEP.join(id_to_path_labels_cb[lid])
                for lid in id_to_path_ids_cb
            ]
            cb_trie = LabelTrie(
                labels=sorted(path_strings_cb),
                tokenizer=processor.tokenizer,
                separator=LABELS_SEP,
            )
        callback = SoftHierarchicalEvalCallback(
            trainer_ref=trainer,
            processor=processor,
            eval_dataset=ds["validation"],
            system_prompt=args.system_prompt,
            trie=cb_trie,
            canonical_to_id=c2i_cb,
            normalized_to_canonical=n2c_cb,
            id_to_path_ids=id_to_path_ids_cb,
            ic_table=ic_table_cb,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.soft_eval_samples,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_beams=args.num_beams,
        )
        trainer.add_callback(callback)
        print(
            f"[CALLBACK] SoftHierarchicalEvalCallback activé "
            f"({args.soft_eval_samples} ex/val, IC={'on' if ic_table_cb else 'off'})"
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

    normalized_to_canonical, canonical_to_id, id_to_label = load_timel_term_reference(
        args.classes_tsv
    )
    print(f"[REF] {len(normalized_to_canonical)} termes valides chargés depuis {args.classes_tsv}")

    # V6 : taxonomie obligatoire si path_mode actif
    id_to_path_ids: Dict[str, List[str]] = {}
    id_to_path_labels: Dict[str, List[str]] = {}
    if args.path_mode:
        if not args.taxonomy_json:
            raise ValueError("--path_mode true requiert --taxonomy_json en mode predict.")
        id_to_path_ids, id_to_path_labels = load_timel_taxonomy(
            args.taxonomy_json, id_to_label
        )
        print(f"[TAXO] {len(id_to_path_ids)} chemins indexés")

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

    # Trie construit une seule fois (réutilisé pour tous les exemples).
    # En V6 path_mode : trie indexé sur les chemins complets, séparateur ' ; '.
    # En V6 flat   : trie indexé sur les feuilles uniquement, séparateur ', ' (= V5).
    trie: Optional[LabelTrie] = None
    if args.constrained_decoding:
        if args.path_mode:
            path_strings = [
                PATH_SEP.join(id_to_path_labels[lid])
                for lid in id_to_path_ids
            ]
            trie = LabelTrie(
                labels=sorted(path_strings),
                tokenizer=processor.tokenizer,
                separator=LABELS_SEP,
            )
            print(
                f"[TRIE/PATH] {len(path_strings)} chemins indexés "
                f"→ {trie.n_nodes} noeuds, profondeur max {trie.max_depth}, "
                f"séparateur {LABELS_SEP!r} = {trie.sep_tokens}"
            )
        else:
            trie = LabelTrie(
                labels=sorted(normalized_to_canonical.values()),
                tokenizer=processor.tokenizer,
            )
            print(
                f"[TRIE] {len(normalized_to_canonical)} labels indexés "
                f"→ {trie.n_nodes} noeuds, profondeur max {trie.max_depth}, "
                f"séparateur ', ' = {trie.sep_tokens}"
            )

    out_path = Path(args.pred_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_valid = 0
    total_invalid = 0
    empty_predictions = 0
    exact_match = 0
    mean_pred_card_sum = 0
    mean_gold_card_sum = 0

    # Compteurs P/R/F1
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    per_class: Dict[str, Dict[str, int]] = {}
    pred_label_counts: Dict[str, int] = {}
    n_with_gold = 0

    # Métriques hiérarchiques (V6) : agrégation au niveau des ancêtres
    h_tp = 0
    h_fp = 0
    h_fn = 0
    n_with_taxo = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            prompt_messages = inject_system_prompt(
                keep_prompt_messages_only(ex["messages"]),
                args.system_prompt,
            )
            prompt_messages = sanitize_messages_for_template(prompt_messages)
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

            prompt_len = batch["input_ids"].shape[1]
            logits_processors = None
            if trie is not None:
                logits_processors = LogitsProcessorList([
                    LabelTrieLogitsProcessor(
                        trie=trie,
                        tokenizer=processor.tokenizer,
                        prompt_len=prompt_len,
                    )
                ])

            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=args.num_beams,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    logits_processor=logits_processors,
                )

            generated_new = generated[:, prompt_len:]
            raw_text = processor.batch_decode(
                generated_new,
                skip_special_tokens=True,
            )[0].strip()

            if args.path_mode:
                normalized = parse_path_output(raw_text, normalized_to_canonical)
            else:
                normalized = normalize_prediction_terms(raw_text, normalized_to_canonical)
            pred_terms = normalized["valid_terms"]
            invalid_terms = normalized["invalid_terms"]
            pred_ids = [canonical_to_id[t] for t in pred_terms]
            gold_terms = extract_gold_terms_from_example(ex, normalized_to_canonical)

            total_valid += len(pred_terms)
            total_invalid += len(invalid_terms)
            if not pred_terms:
                empty_predictions += 1

            mean_pred_card_sum += len(pred_terms)
            mean_gold_card_sum += len(gold_terms)

            for t in pred_terms:
                pred_label_counts[t] = pred_label_counts.get(t, 0) + 1

            if gold_terms:
                n_with_gold += 1
                pset, gset = set(pred_terms), set(gold_terms)
                if pset == gset:
                    exact_match += 1
                tp = len(pset & gset)
                fp = len(pset - gset)
                fn = len(gset - pset)
                micro_tp += tp
                micro_fp += fp
                micro_fn += fn
                for t in pset & gset:
                    pc = per_class.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                    pc["tp"] += 1
                for t in pset - gset:
                    pc = per_class.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                    pc["fp"] += 1
                for t in gset - pset:
                    pc = per_class.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                    pc["fn"] += 1

                # H-F1 sur l'union des ancêtres (V6)
                if id_to_path_ids:
                    pred_anc: Set[str] = set()
                    for label in pset:
                        tid = canonical_to_id.get(label)
                        if tid and tid in id_to_path_ids:
                            pred_anc |= set(id_to_path_ids[tid])
                    gold_anc: Set[str] = set()
                    for label in gset:
                        tid = canonical_to_id.get(label)
                        if tid and tid in id_to_path_ids:
                            gold_anc |= set(id_to_path_ids[tid])
                    if pred_anc or gold_anc:
                        n_with_taxo += 1
                        h_tp += len(pred_anc & gold_anc)
                        h_fp += len(pred_anc - gold_anc)
                        h_fn += len(gold_anc - pred_anc)

            record = {
                "index": idx,
                "image": ex["images"][0],
                "raw_prediction": raw_text,
                "predicted_terms": pred_terms,
                "predicted_ids": pred_ids,
                "invalid_terms": invalid_terms,
            }
            if gold_terms:
                record["gold_terms"] = gold_terms
                record["gold_ids"] = [canonical_to_id[t] for t in gold_terms if t in canonical_to_id]

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(ds)

    def _f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    micro_p, micro_r, micro_f1 = _f1(micro_tp, micro_fp, micro_fn)

    macro_p_sum = 0.0
    macro_r_sum = 0.0
    macro_f1_sum = 0.0
    macro_n = 0
    for _label, c in per_class.items():
        if (c["tp"] + c["fn"]) == 0:
            continue  # ignore les classes jamais en gold
        p, r, fmes = _f1(c["tp"], c["fp"], c["fn"])
        macro_p_sum += p
        macro_r_sum += r
        macro_f1_sum += fmes
        macro_n += 1
    macro_p = macro_p_sum / macro_n if macro_n > 0 else 0.0
    macro_r = macro_r_sum / macro_n if macro_n > 0 else 0.0
    macro_f1 = macro_f1_sum / macro_n if macro_n > 0 else 0.0

    total_emitted = total_valid + total_invalid
    halluc_rate = (total_invalid / total_emitted) if total_emitted > 0 else 0.0
    exact_match_rate = (exact_match / n_with_gold) if n_with_gold > 0 else 0.0

    top20 = sorted(pred_label_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top20_out = [
        [lab, cnt, (cnt / n) if n > 0 else 0.0] for lab, cnt in top20
    ]

    metrics = {
        "n_examples": n,
        "n_with_gold": n_with_gold,
        "constrained_decoding": bool(args.constrained_decoding),
        "path_mode": bool(args.path_mode),
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "macro_classes": macro_n,
        "exact_match_rate": exact_match_rate,
        "mean_pred_card": (mean_pred_card_sum / n) if n > 0 else 0.0,
        "mean_gold_card": (mean_gold_card_sum / n) if n > 0 else 0.0,
        "hallucination_rate": halluc_rate,
        "empty_predictions": empty_predictions,
        "total_valid_terms": total_valid,
        "total_invalid_terms": total_invalid,
        "top20_predicted_labels": top20_out,
    }

    if id_to_path_ids:
        h_p, h_r, h_f1 = _f1(h_tp, h_fp, h_fn)
        metrics.update({
            "n_with_taxo": n_with_taxo,
            "hierarchical_micro_p": h_p,
            "hierarchical_micro_r": h_r,
            "hierarchical_micro_f1": h_f1,
        })

    metrics_path = Path(str(out_path) + ".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fm:
        json.dump(metrics, fm, ensure_ascii=False, indent=2)

    print(f"[PREDICT] Prédictions sauvegardées dans: {out_path}")
    print(f"[PREDICT] Métriques sauvegardées dans: {metrics_path}")
    print(
        f"[PREDICT] n={n} n_with_gold={n_with_gold} "
        f"micro_f1={micro_f1:.4f} macro_f1={macro_f1:.4f} "
        f"exact={exact_match_rate:.3f} halluc={halluc_rate:.3f} empty={empty_predictions}"
    )
    if id_to_path_ids:
        h_p, h_r, h_f1 = _f1(h_tp, h_fp, h_fn)
        print(
            f"[PREDICT/HIER] micro_p={h_p:.4f} micro_r={h_r:.4f} "
            f"micro_f1={h_f1:.4f}"
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
