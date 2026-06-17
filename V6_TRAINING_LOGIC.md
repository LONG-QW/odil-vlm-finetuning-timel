# V6 — Pipeline d'entraînement Qwen3-VL pour la classification iconographique TIMEL

Document de référence pour la rédaction d'article. Décrit le pipeline V6 du projet ODIL : fine-tuning supervisé d'un modèle vision-langage Qwen3-VL sur l'annotation iconographique multi-label avec le référentiel TIMEL (Thésaurus Iconographique de Manuscrits Enluminés Liturgiques).

---

## 0. Patch V6.1 — fixes anti-collapse et anti-sous-prédiction

Suite à inspection des premières prédictions (n=20, mean_pred_card=2.3, gold=11.75, halluc=29%), 6 fixes ont été ajoutés :

1. **Filtrage trie aux feuilles annotables** (P0) : `load_timel_taxonomy` ne retient que les chemins dont la feuille terminale est dans `classes.tsv`. Empêche le modèle d'émettre des nœuds intermédiaires comme `"objet architecture"` seul.
2. **`MinLabelsLogitsProcessor`** (P0) : masque EOS / `<|im_end|>` tant que `(min_labels - 1)` séparateurs ` ; ` n'ont pas été émis. Active via `--min_labels 5`.
3. **Beam search par défaut** : `--num_beams 4` au lieu de 1.
4. **Pénalités** : `--repetition_penalty 1.0` (avant 1.1 qui induisait un EOS précoce) + `--length_penalty 1.2` (favorise séquences plus longues sous beam).
5. **Class-balanced sampling** : `--class_balanced true --balance_factor 2.0` rééchantillonne le train avec remplacement, poids = max IC des labels de l'exemple → classes rares sur-représentées.
6. **Tri des cibles par profondeur** : dans le collator, `canon_leaves.sort(key=(len(path), label))`. Stabilise l'ordre d'émission entre exemples (chemins courts en premier puis alphabétique), accélère la convergence sur les préfixes partagés.

---

## 1. Tâche et données

### 1.1 Tâche

Classification multi-label d'enluminures médiévales avec le référentiel TIMEL :
- **Entrée** : une image d'enluminure
- **Sortie** : un ensemble d'étiquettes iconographiques (description du contenu : personnages, scènes, objets, thèmes)
- **Cardinalité observée** : 5 à 35 étiquettes par image (moyenne ~10)

### 1.2 Référentiel TIMEL

- **2 333 feuilles annotables** (vocabulaire fermé) — fichier `classes.tsv` (colonnes `timel_id`, `timel_label`)
- **Taxonomie hiérarchique** dans `timel_taxonomy.json` : 4 660 nœuds, 2 243 nœuds intermédiaires non annotables directement, 5 racines (`nature lieu`, `objet architecture`, `personnage`, `sujet`, `thème`)
- **Profondeur** : 1 à 8 niveaux, moyenne 4
- **Format des labels** : libellés canoniques français (ex. `"hérisson"`, `"Joseph interprétant le songe du panetier"`)

### 1.3 Dataset

- **train_terms.jsonl** : 8 676 exemples annotés
- Format messages (JSONL une ligne par exemple) :
  ```json
  {
    "images": ["data/images/006989.jpg"],
    "messages": [
      {"role": "user", "content": [
        {"type": "image", "image": "data/images/006989.jpg"},
        {"type": "text", "text": "Générer les mots-clés timel pour cette image."}
      ]},
      {"role": "assistant", "content": [
        {"type": "text", "text": "saint François, sacrifice, Osée, Joseph, ..."}
      ]}
    ]
  }
  ```

---

## 2. Pipeline V6

### 2.1 Modèle

- **Modèle de base** : `Qwen/Qwen3-VL-4B-Instruct`
- **Architecture** : vision-language model (vision tower + projection + LLM décodeur autorégressif)
- **Précision** : bf16
- **Gradient checkpointing** activé (use_reentrant=False)

### 2.2 Supervision *path-based*

Innovation principale de V6 : au lieu d'entraîner le modèle à produire la liste plate des feuilles, on l'entraîne à produire le **chemin hiérarchique complet** depuis la racine taxonomique jusqu'à chaque feuille.

**Sérialisation** :
- Séparateur intra-chemin : ` / `
- Séparateur inter-chemins : ` ; `

**Exemple** (cible reformulée) :

| Format V5 (flat) | Format V6 (path-based) |
|---|---|
| `hérisson, château` | `nature lieu / animaux / animaux terrestres / hérisson ; objet architecture / éléments d'architecture / château` |

**Justification** : les nœuds intermédiaires (`nature lieu`, `animaux`) sont fréquents et partagés par de nombreuses feuilles. Leur entraînement fournit un signal stable pour les feuilles rares (longue traîne). Le partage de préfixes au niveau token donne une compositionnalité comparable à celle d'Iconclass (`11H(JOSEPH)32`).

La réécriture est faite **en ligne** dans le collator (`rewrite_messages_to_paths`) à partir des données plates de `train_terms.jsonl` + `timel_taxonomy.json`. Pas de pré-traitement.

### 2.3 System prompt fixe

Injecté à la fois en entraînement et en inférence (cohérence prompt train/inférence) :

```
Tu es un classifieur visuel iconographique. Liste les étiquettes TIMEL
pertinentes pour cette image SOUS FORME DE CHEMIN HIÉRARCHIQUE COMPLET,
depuis la racine taxonomique jusqu'à la feuille. Utilise ' / ' entre les
niveaux et ' ; ' entre étiquettes distinctes. N'écris aucun commentaire,
aucun doublon, et utilise exactement les libellés canoniques. Termine ta
sortie après la dernière feuille.
```

### 2.4 Loss et masquage

- **Loss** : cross-entropy au niveau token, agrégée uniquement sur les tokens de la réponse `assistant`.
- **Masquage** : tous les tokens de prompt, système, user, et chat-template sont mis à `-100` dans les `labels`. La position de début de masquage est calculée via `build_assistant_header_ids` qui repère la séquence `<|im_start|>assistant\n` comme **séquence d'IDs** (et non comme chaîne tokenisée brute, ce qui était un bug critique dans V2).

### 2.5 Décodage contraint par trie *path-based*

À l'inférence, un `LogitsProcessor` filtre les logits à chaque étape pour ne laisser passer que les tokens menant à une continuation valide dans le trie construit sur les 2 333 chemins complets.

- **Construction du trie** : pour chaque feuille `f`, on tokenise sa chaîne `racine / ... / f` (variante "fresh" et variante "après séparateur" pour absorber l'influence du leading-space des tokenizers BPE). Chaque chemin = chaîne d'IDs token = chemin dans le trie.
- **Transitions** :
  - À la racine : tout premier-token d'un chemin valide
  - Mid-trie : enfants du nœud courant
  - Nœud terminal : (a) début du séparateur ` ; ` pour démarrer un nouveau chemin, (b) `<|im_end|>` ou EOS pour clore
- **Propriétés garanties** :
  - Pas de paraphrase (sortie ∈ vocabulaire fini)
  - Pas de troncature (le trie connaît la longueur des chemins)
  - Pas de doublon (via `no_repeat_ngram_size`)
  - Pas d'hallucination de chemins inexistants

### 2.6 Hyperparamètres recommandés

| Paramètre | Valeur | Justification |
|---|---|---|
| `max_steps` | 10 000 | Convergence raisonnable sur 8 676 exemples avec `grad_accum=8`, `bs=1` |
| `learning_rate` | 2e-5 | Standard SFT VLM |
| `per_device_batch_size` | 1 | Contrainte VRAM (4B params + image features) |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8 |
| `lr_scheduler_type` | `cosine` | Standard |
| `warmup_ratio` | 0.03 | Standard |
| `weight_decay` | 0.05 | Standard |
| `precision` | `bf16` | A40/A100/H100 |
| `max_pixels` | 512×28×28 (~401k) | Contrôle de la résolution image, évite OOM |
| `max_new_tokens` (inférence) | 1 280 | Sortie path-based ~4-6× plus longue que flat |
| `repetition_penalty` | 1.1 | Décourage la répétition de tokens (collapse) |
| `no_repeat_ngram_size` | 5 | Interdit la répétition exacte de 5-grams |
| `num_beams` | 1 (greedy) ; 4 si possible | Beam search améliore le multi-label |
| `eval_strategy` | `steps` | Eval pendant le training |
| `eval_steps` | 200 | Compromis qualité/coût |

---

## 3. Métriques

### 3.1 Flat multi-label

- **Micro P/R/F1** : agrégation TP/FP/FN sur tous les exemples ; binaire par classe ; le F1 dominant est celui qu'on a en tête quand on parle de "performance multi-label".
- **Macro P/R/F1** : moyenne des F1 par classe (uniquement classes vues en gold). Très sensible à la longue traîne.
- **Exact-match rate** : proportion d'exemples où `pred_set == gold_set`.
- **Hallucination rate** : `n_invalid / (n_valid + n_invalid)`. Avec décodage contraint, devrait être ≈ 0.

### 3.2 Hierarchical micro P/R/F1

Crédit partiel pour erreurs intra-branche. Définition à la Kiritchenko / Verspoor :

Soit `Anc(c)` l'ensemble des ancêtres de `c` (incluant `c` lui-même). Pour un exemple avec preds `P` et golds `G` :

- `pred_anc = ⋃_{p ∈ P} Anc(p)`
- `gold_anc = ⋃_{g ∈ G} Anc(g)`
- `H-TP = |pred_anc ∩ gold_anc|`, `H-FP = |pred_anc − gold_anc|`, `H-FN = |gold_anc − pred_anc|`
- Agrégation micro → `H-P`, `H-R`, `H-F1`

Effet : prédire `anguille` alors que le gold est `saumon` donne un crédit non-nul car `animaux aquatiques` est dans les deux ensembles d'ancêtres.

### 3.3 Soft hierarchical similarity (nôtre)

Similarité douce entre deux nœuds :

> $\mathrm{sim}_{\mathrm{soft}}(a, b) = \dfrac{\mathrm{depth}(\mathrm{LCA}(a, b))}{\max(\mathrm{depth}(a), \mathrm{depth}(b))}$

**Convention** : racine de profondeur 0. `depth(node) = len(path_ids) - 1`.

**Propriétés** :
- $0 \leq \mathrm{sim}_{\mathrm{soft}}(a, b) \leq 1$
- $\mathrm{sim}_{\mathrm{soft}}(a, a) = 1$
- Si $a$ et $b$ ne partagent que la racine, $\mathrm{sim}_{\mathrm{soft}}(a, b) = 0$
- Deux frères de profondeur $d$ : $\mathrm{sim}_{\mathrm{soft}} = \dfrac{d-1}{d}$ (ex. $d=4 \Rightarrow 0{,}75$)

**Soft P / R / F1** (multi-label) :
- $P = \dfrac{1}{|P|} \sum_{p \in P} \max_{g \in G} \mathrm{sim}_{\mathrm{soft}}(p, g)$
- $R = \dfrac{1}{|G|} \sum_{g \in G} \max_{p \in P} \mathrm{sim}_{\mathrm{soft}}(g, p)$
- $F_1 = \dfrac{2PR}{P + R}$

Agrégation macro sur exemples (moyenne).

**Différence avec H-F1** : H-F1 mesure le recouvrement *ensembliste* d'ancêtres ; sim_soft mesure une *similarité graduelle continue* entre paires de feuilles, indépendamment du contexte multi-label. Les deux sont complémentaires.

### 3.4 IC-weighted soft P/R/F1

Pondération par l'Information Content de chaque label, calculé sur le train avec lissage additif (Laplace) :

> $P(c) = \dfrac{\mathrm{count}(c) + \alpha}{N + \alpha \cdot |V|}, \quad IC(c) = -\log P(c)$

Avec `count(c)` = doc-frequency du label dans `train_terms.jsonl`, `N` = nombre de documents, `|V|` = taille du vocabulaire (= 2 333), $\alpha = 1$.

**Exemple typique** (N = 8 661, |V| = 2 333, α = 1) :
- `count(animal_freq) = 3 000` → $P \approx 0{,}27$ → $IC \approx 1{,}3$
- `count(label_rare) = 10` → $P \approx 0{,}0009$ → $IC \approx 7{,}0$

**IC-weighted P** :

$$P_{IC} = \frac{\sum_{p \in P} IC(p) \cdot \max_{g \in G} \mathrm{sim}_{\mathrm{soft}}(p, g)}{\sum_{p \in P} IC(p)}$$

(et symétriquement pour R)

**Effet** : les erreurs sur les classes rares (IC élevé) sont plus pénalisantes. Une bonne prédiction d'un label rare est davantage récompensée. C'est l'analogue continu d'une F1 macro mais qui exploite la similarité hiérarchique.

---

## 4. Commandes

### 4.1 Préparation

Pré-requis : fichiers `train_terms.jsonl`, `val_terms.jsonl`, `classes.tsv`, `timel_taxonomy.json` colocalisés.

### 4.2 Training V6 (avec patch V6.1)

```bash
python train_qwen3vl_sft_timel_merged_fr_V6.py \
  --mode train \
  --train_jsonl train_terms.jsonl \
  --val_jsonl val_terms.jsonl \
  --classes_tsv classes.tsv \
  --taxonomy_json timel_taxonomy.json \
  --path_mode true \
  --output_dir qwen3_vl_timel_V6_out \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --max_steps 10000 \
  --lr 2e-5 \
  --per_device_bs 1 \
  --grad_accum 8 \
  --warmup_ratio 0.03 \
  --weight_decay 0.05 \
  --lr_scheduler_type cosine \
  --precision bf16 \
  --gradient_checkpointing true \
  --logging_steps 20 \
  --save_steps 200 \
  --eval_steps 200 \
  --save_total_limit 2 \
  --seed 42 \
  --class_balanced true --balance_factor 2.0 --ic_alpha 1.0 \
  --num_beams 4 --repetition_penalty 1.0 --length_penalty 1.2 \
  --min_labels 5 \
  --soft_eval_samples 100 --soft_eval_ic true
```

Le `--soft_eval_samples 100` active le `SoftHierarchicalEvalCallback` qui génère sur 100 exemples du val à chaque `eval_steps` et loggue `eval_soft_f1`, `eval_ic_soft_f1` (visible dans logs / TensorBoard).

### 4.3 Inférence V6 (avec patch V6.1)

```bash
python train_qwen3vl_sft_timel_merged_fr_V6.py \
  --mode predict \
  --model_name qwen3_vl_timel_V6_out/final \
  --predict_jsonl val_terms.jsonl \
  --classes_tsv classes.tsv \
  --taxonomy_json timel_taxonomy.json \
  --path_mode true \
  --constrained_decoding true \
  --pred_out results-V6.jsonl \
  --max_new_tokens 1280 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --length_penalty 1.2 \
  --no_repeat_ngram_size 5 \
  --min_labels 5
```

Output :
- `results-V6.jsonl` : ligne par exemple avec `predicted_terms`, `predicted_ids`, `gold_terms`, `gold_ids`, `raw_prediction`
- `results-V6.jsonl.metrics.json` : flat + hierarchical metrics

### 4.4 Évaluation offline complète (avec soft + IC)

```bash
python eval_timel_predictions.py \
  --pred_jsonl results-V6.jsonl \
  --classes_tsv classes.tsv \
  --gold_jsonl val_terms.jsonl \
  --taxonomy_json timel_taxonomy.json \
  --train_jsonl train_terms.jsonl \
  --alpha 1.0 \
  --out_json results-V6.full_metrics.json \
  --per_class_csv results-V6.per_class.tsv \
  --top_k 20
```

Sortie additionnelle :
- `soft_p`, `soft_r`, `soft_f1` (macro sur exemples)
- `ic_soft_p`, `ic_soft_r`, `ic_soft_f1`
- `hierarchical_micro_p/r/f1`
- Rapport par classe trié par F1

---

## 5. Reproductibilité

### 5.1 Seeds et déterminisme

`--seed 42` (graine PyTorch + Transformers + dataset shuffle).

### 5.2 Versions logicielles

À enregistrer au moment du run :

```bash
python - <<'PY'
import torch, transformers, trl, datasets, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("datasets", datasets.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

### 5.3 GPU / mémoire

Profil typique observé avec Qwen3-VL-4B + bf16 + gradient checkpointing + max_pixels=512×28×28 :
- VRAM allouée : ~20–25 GB (A40 48 GB ample, A100 40 GB compatible)
- Temps par step : ~3–5 s (avec eval callback générant sur 100 ex, overhead ~5 min par éval)

---

## 6. Notes pour le papier

### 6.1 Comparaisons

- **Baseline iconclass-vlm** : `small-models-for-glam/iconclass-vlm` (Qwen2.5-VL-3B SFT sur 87 744 images Brill Iconclass). Différences clés :
  - Taxonomie : Iconclass compositionnelle par notation (`11H(JOSEPH)32`) ; TIMEL compositionnelle par langage naturel (libellés français)
  - Échelle : ~9× plus grand
  - Pas de métrique rapportée → comparaison qualitative uniquement
- **V4 (IDs opaques)** : échec total (micro_f1 = 0, hallucination = 100 %)
- **V5 (labels flat)** : baseline raisonnable, expose le mode-collapse sur la longue traîne
- **V6 (path-based + soft metric)** : exploite la hiérarchie

### 6.2 Sources à citer

- Qwen3-VL : famille de modèles Alibaba
- TIMEL : Thésaurus Iconographique (CESCM / ODIL, école nationale des chartes)
- TRL : von Werra et al., 2020 (citation BibTeX dans la card iconclass-vlm)
- Hierarchical evaluation : Kiritchenko et al., 2005 ("Functional annotation of genes using hierarchical text categorization") ; Verspoor et al., 2006
- Information Content : Resnik 1995 (sémantique distributionnelle + IC)
- Décodage contraint par trie : approche standard pour génération à vocabulaire contrôlé (préfixes / outlines / lm-format-enforcer)

### 6.3 Ablations à reporter

- V5 (flat) vs V6 (path-based) à hyperparamètres identiques
- V6 sans décodage contraint vs V6 avec décodage contraint
- V6 avec / sans IC pondération
- Greedy (num_beams=1) vs beam search (num_beams=4)
- Effet de `max_pixels` sur la qualité

### 6.4 Contributions revendiquées

1. Pipeline VLM SFT path-based pour vocabulaire iconographique à 2 333 classes longue traîne avec compositionnalité naturelle exploitée
2. Décodage contraint par trie path-based éliminant paraphrase, hallucination, troncature, doublon
3. Métrique *soft hierarchical similarity* avec pondération IC
4. Comparaison avec baseline iconclass-vlm sur GLAM (Galleries, Libraries, Archives, Museums)
