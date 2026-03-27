# Supervised-Learning-Project (NLP Project 2)

Ce projet est dédié au traitement du langage naturel (NLP) appliqué à des avis clients du domaine de l'assurance. Il propose un pipeline complet de bout en bout : nettoyage des données, exploration non supervisée (Topic Modeling via NMF, plongements de mots Word2Vec et GloVe), une approche supervisée pour l'analyse de sentiment (TF-IDF, Keras, Transformers HuggingFace), ainsi qu'une application web interactive (Streamlit) pour la prédiction, la recherche sémantique, la détection de thèmes, les résumés NLP et le Question-Answering (RAG).

L'entraînement des modèles a été effectué sur un dataset français, merci d'utiliser des avis clients en français pour faciliter l'utilisation.

---

## 1. Dépendances & Bibliothèques

Python 3.9+ requis. Bibliothèques principales :

| Bibliothèque | Usage |
|---|---|
| `transformers` (HuggingFace) | Zero-shot, Résumé (T5), QA (CamemBERT) |
| `tensorflow` / `tf-keras` | Modèles deep learning avec Embedding layer |
| `gensim` | Word2Vec + GloVe (downloader API) |
| `scikit-learn` | TF-IDF, Logistic Regression, NMF Topic Modeling |
| `streamlit` | Application web interactive |
| `deep-translator` | Traduction FR → EN (colonne `avis_en`) |
| `sentencepiece` | Tokenizer T5 (requis par le modèle de résumé) |
| `pyspellchecker` | Correction orthographique |
| `pandas`, `numpy`, `matplotlib` | Traitement des données & visualisation |
| `openpyxl` | Lecture des fichiers `.xlsx` sources |
| `torch` | Backend PyTorch pour les modèles HuggingFace |
| `tensorboard` | Visualisation des embeddings et des courbes d'entraînement |

---

## 2. Installation de l'environnement

```bash
# 1. Créer le venv (si pas déjà fait)
python -m venv .venv

# 2. Activer le venv
# Windows PowerShell :
.\.venv\Scripts\Activate.ps1
# Windows CMD :
.\.venv\Scripts\activate.bat
# Mac/Linux :
source .venv/bin/activate

# 3. Installer toutes les dépendances
pip install -r requirements.txt
```

---

## 3. Pipeline complet — ordre d'exécution

### Phase 1 — Préparation & Nettoyage des données
```bash
python src/01_explore_data.py      # Exploration initiale des fichiers .xlsx
python src/02_build_dataset.py     # Fusion des fichiers en full_dataset.csv
python src/03_data_cleaning.py     # Normalisation, correction orthographique, mots fréquents & bigrammes
```

### Phase 2 — Résumé & Traduction
```bash
python src/summary_translation.py  # Génère insurer_summaries.csv + colonne avis_en (FR→EN) dans le dataset
```
> ⚠️ Cette étape nécessite `sentencepiece` et `deep-translator`. Elle peut prendre plusieurs minutes (téléchargement du modèle T5 si absent du cache).

### Phase 3 — Modélisation non-supervisée & Embeddings
```bash
python src/04_topic_modeling.py       # Topic Modeling NMF → topics.txt
python src/05_word_embeddings.py      # Word2Vec → word2vec.model + PCA + export TensorBoard
python src/05b_glove_embeddings.py    # GloVe (via gensim downloader) → glove_pca.png + export TensorBoard
```

### Phase 4 — Apprentissage supervisé
```bash
python src/06_supervised_learning_baseline.py  # TF-IDF + Logistic Regression → .pkl
python src/07_hf_transformer_model.py          # Zero-shot BERT (nlptown) → évaluation sur échantillon
python src/08_keras_basic_embedding.py         # Keras Embedding aléatoire + TensorBoard → .keras
python src/09_keras_pretrained_embedding.py    # Keras Embedding Word2Vec pré-entraîné + TensorBoard → .keras
```

### Phase 5 — Analyse & Prédiction de thèmes
```bash
python src/10_error_analysis.py                # Analyse des erreurs (FP/FN) → error_analysis.txt
python src/11_category_stars_prediction.py     # Détection de thèmes Zero-Shot → sample_categories.csv
```

---

## 4. Visualisation TensorBoard

```bash
tensorboard --logdir logs/fit
# Ouvrir : http://localhost:6006
```
Les fichiers `.tsv` de Word2Vec et GloVe peuvent aussi être chargés sur [projector.tensorflow.org](https://projector.tensorflow.org).

---

## 5. Lancer l'application Streamlit

Une fois les étapes ci-dessus exécutées :
```bash
streamlit run src/app.py
```

### Fonctionnalités de l'app (5 onglets) :

| Onglet | Fonctionnalité |
|---|---|
| **Sentiment & Explication** | Prédiction TF-IDF + poids des mots + comparaison de tous les modèles |
| **Analyse des assureurs** | Moyenne des notes, distribution des sentiments, résumés NLP par assureur, recherche par mot-clé |
| **Recherche Sémantique** | Requête Word2Vec avec mots similaires + filtrage des avis correspondants |
| **Détection de catégories** | Classification Zero-Shot live (tarifs, service client, remboursement, etc.) |
| **Résumé & QA (RAG)** | Résumé automatique des avis d'un assureur + Question-Réponse extractif sur les avis |
