# Supervised-Learning-Project (NLP Project 2)

This project applies Natural Language Processing (NLP) to French customer reviews in the insurance sector. It implements a complete end-to-end pipeline: data cleaning, unsupervised exploration (NMF Topic Modeling, Word2Vec and GloVe embeddings), supervised sentiment analysis (TF-IDF, Keras, HuggingFace Transformers), and an interactive Streamlit web application for prediction, semantic search, topic detection, NLP summarization and Question-Answering (RAG).

All models were trained on a French-language dataset — please use French customer reviews for best results.

---

## 1. Dependencies & Libraries

Python 3.9+ required. Main libraries:

| Library | Usage |
|---|---|
| `transformers` (HuggingFace) | Zero-shot classification, Summarization (T5), QA (CamemBERT) |
| `tensorflow` / `tf-keras` | Deep learning models with Embedding layers |
| `gensim` | Word2Vec + GloVe (downloader API) |
| `scikit-learn` | TF-IDF, Logistic Regression, NMF Topic Modeling |
| `streamlit` | Interactive web application |
| `deep-translator` | Translation FR → EN (`avis_en` column) |
| `sentencepiece` | T5 tokenizer (required by the summarization model) |
| `pyspellchecker` | Spell correction |
| `pandas`, `numpy`, `matplotlib` | Data processing & visualization |
| `openpyxl` | Reading source `.xlsx` files |
| `torch` | PyTorch backend for HuggingFace models |
| `tensorboard` | Embedding visualization and training curve monitoring |

---

## 2. Environment Setup

```bash
# 1. Create the virtual environment (if not already done)
python -m venv .venv

# 2. Activate the venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat
# Mac/Linux:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## 3. Full Pipeline — Execution Order

### Phase 1 — Data Preparation & Cleaning
```bash
python src/01_explore_data.py      # Initial exploration of .xlsx files
python src/02_build_dataset.py     # Merge files into full_dataset.csv
python src/03_data_cleaning.py     # Normalization, spell correction, frequent words & bigrams
```

### Phase 2 — Summarization & Translation
```bash
python src/summary_translation.py  # Generates insurer_summaries.csv + avis_en column (FR→EN)
```
> ⚠️ This step requires `sentencepiece` and `deep-translator`. It may take several minutes (downloads T5 model if not cached).

### Phase 3 — Unsupervised Modeling & Embeddings
```bash
python src/04_topic_modeling.py       # NMF Topic Modeling → topics.txt
python src/05_word_embeddings.py      # Word2Vec → word2vec.model + PCA + TensorBoard export
python src/05b_glove_embeddings.py    # GloVe (via gensim downloader) → glove_pca.png + TensorBoard export
```

### Phase 4 — Supervised Learning
```bash
python src/06_supervised_learning_baseline.py  # TF-IDF + Logistic Regression → .pkl
python src/07_hf_transformer_model.py          # Zero-shot BERT (nlptown) → evaluation on sample
python src/08_keras_basic_embedding.py         # Keras random Embedding + TensorBoard → .keras
python src/09_keras_pretrained_embedding.py    # Keras Word2Vec pre-trained Embedding + TensorBoard → .keras
```

### Phase 5 — Analysis & Topic Prediction
```bash
python src/10_error_analysis.py                # Error analysis (FP/FN) → error_analysis.txt
python src/11_category_stars_prediction.py     # Zero-Shot topic detection → sample_categories.csv
```

---

## 4. TensorBoard Visualization

```bash
tensorboard --logdir logs/fit
# Open: http://localhost:6006
```
Word2Vec and GloVe `.tsv` files can also be loaded on [projector.tensorflow.org](https://projector.tensorflow.org).

---

## 5. Launch the Streamlit App

Once all pipeline steps have been run:
```bash
streamlit run src/app.py
```

### App Features (5 tabs):

| Tab | Feature |
|---|---|
| **Sentiment & Explanation** | TF-IDF prediction + word importance + comparison of all 4 models |
| **Insurer Analysis** | Average rating per insurer, sentiment distribution, NLP summaries, keyword search |
| **Semantic Search** | Word2Vec query with similar words + filtering of matching reviews |
| **Category Detection** | Live Zero-Shot classification (pricing, customer service, reimbursement, etc.) |
| **Summary & QA (RAG)** | Automatic summary of an insurer's reviews + extractive Question-Answering |
