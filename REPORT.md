# Technical Report — Supervised NLP Project
### Customer Review Analysis in the Insurance Sector

---

## 1. Context & Data

### Dataset
The dataset consists of French-language customer reviews about insurance companies (car, health, home insurance). The raw data is contained in several `.xlsx` files merged into a single CSV (`data/raw/full_dataset.csv`). Each row contains:
- `assureur`: the company name
- `avis`: the review text (in French)
- `note`: the rating given (1 to 5 stars)

---

## 2. Data Exploration & Cleaning

### Technical choices (`03_data_cleaning.py`)
- **Normalization**: lowercasing, removal of line breaks and extra whitespace.
- **Spell correction**: implemented via `pyspellchecker` using a cache-optimized approach (top 2000 unknown words identified corpus-wide, corrected once). This reduces execution time from several hours to a few minutes.
- **Sentiment labeling**: the `sentiment` column is derived from the rating:
  - Rating ≥ 4 → `positive`
  - Rating ≤ 2 → `negative`
  - Rating = 3 → `neutral`

### Frequent words and bigrams
The 20 most frequent words in the corpus (after cleaning) reflect French grammatical structure. Bigrams are more informative and reveal satisfaction patterns:

| Bigram | Occurrences | Interpretation |
|---|---|---|
| `je suis` | 11,457 | Beginning of a stance expression |
| `suis satisfait` | 2,812 | Strong satisfaction signal |
| `le service` | 2,129 | Frequent reference to customer service |
| `les prix` | 2,108 | Sensitive topic of pricing |
| `je recommande` | 1,896 | Explicit recommendation |
| `en charge` | 2,093 | Claims / file management |

**Conclusion**: The corpus revolves around 3 main axes — overall satisfaction, pricing, and customer service/claims. This matches perfectly the Zero-Shot categories defined in the topic detection task.

---

## 3. Topic Modeling (`04_topic_modeling.py`)

### Method: NMF (Non-negative Matrix Factorization)
Applied on a TF-IDF matrix (max 1000 features, French stop words), with 5 components.

### Results:

| Topic | Keywords | Interpretation |
|---|---|---|
| **Topic 1** | sinistre, contrat, mois, depuis, sans | Claims management / disputes |
| **Topic 2** | satisfait, prix, service, qualité, client | Satisfaction and value for money |
| **Topic 3** | direct, auto, voiture, véhicule, cher | Car insurance |
| **Topic 4** | rapide, simple, efficace, site, souscription | Digital experience / online sign-up |
| **Topic 5** | téléphone, personne, reçu, jamais | Difficulty reaching customer service |

**Interpretation**: The 5 topics correspond to well-identified axes in the insurance domain: claims, satisfaction, automobile, digital, and customer service.

---

## 4. Word Embeddings

### Word2Vec (`05_word_embeddings.py`)
- **Architecture**: Skip-gram, 100 dimensions, window=5, min_count=5
- **Similarity results**:
  - Words close to `prix`: tarif, cher, augmentation, cotisation → semantic coherence on pricing topic
  - Euclidean distance between `prix` and `tarif` computed and saved
- **Visualization**: 2D PCA of the top 200 words (`word2vec_pca.png`) + TensorBoard export (tensors.tsv / metadata.tsv for 5000 words)

### GloVe (`05b_glove_embeddings.py`)
- **Model**: `glove-twitter-25` loaded via `gensim.downloader` (104.8 MB)
- **Similarity results** (in English):
  - `price` → card, stock, includes, discount, limited
  - `insurance` → exchange, mortgage, transportation, banking
- **Visualization**: 2D PCA of words 100–300 (`glove_pca.png`) + TensorBoard export (glove_tensors.tsv / glove_metadata.tsv)

**W2V vs GloVe comparison**: Word2Vec trained on the French corpus is more thematically relevant (captures `prix`, `remboursement`, etc. in an insurance context). The pre-trained GloVe model on Twitter offers less specialized similarities but broader vocabulary coverage.

---

## 5. Supervised Learning — Sentiment Analysis

### Task: 3-class classification (positive / neutral / negative)

### 5.1 Baseline TF-IDF + Logistic Regression (`06_supervised_learning_baseline.py`)
- **Vectorization**: TF-IDF, max 5000 features, class_weight='balanced'
- **Split**: 80/20, random_state=42

**Results:**
```
Accuracy : 0.7525
               precision  recall  f1-score  support
  negative       0.87      0.85     0.86     2171
   neutral       0.28      0.37     0.32      682
  positive       0.85      0.78     0.81     1968
```
**Analysis**: The neutral class is very hard to predict (f1=0.32). This is explained by its ambiguous definition (rating=3) and lower sample count (682 vs ~2000 for the other classes).

---

### 5.2 Zero-Shot Transformers (`07_hf_transformer_model.py`)
- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment` (multilingual BERT, predicts 1–5 stars, converted to 3 classes)
- **Evaluation**: 200 samples (CPU constraint)

**Results:**
```
Accuracy : 0.7800 (on 200 samples)
  negative : f1 = 0.88   positive : f1 = 0.84   neutral : f1 = 0.24
```
**Analysis**: Comparable performance to the TF-IDF baseline, slightly better overall, but the neutral class remains problematic (f1=0.24). The model "sees" the stars directly, which is a conceptual advantage.

---

### 5.3 Neural Network — Random Embedding (`08_keras_basic_embedding.py`)
- **Architecture**: `Embedding(5000, 50)` → `GlobalAveragePooling1D` → `Dense(24, relu)` → `Dropout(0.5)` → `Dense(3, softmax)`
- **Training**: 10 epochs, Adam, sparse_categorical_crossentropy
- **TensorBoard**: logs saved in `logs/fit/basic_embedding_*`

**Results:**
```
Accuracy : 0.7957
  negative : f1 = 0.86   positive : f1 = 0.85   neutral : f1 = 0.09
```
**Analysis**: Best overall accuracy (+4pp vs baseline), but the neutral class collapses (f1=0.09). The model converged on the two dominant classes.

---

### 5.4 Neural Network — Pre-trained Word2Vec Embedding (`09_keras_pretrained_embedding.py`)
- **Architecture**: Same as model 5.3, but the Embedding layer is initialized with Word2Vec weights (100 dim, frozen)
- **Training**: 15 epochs, Adam

**Results:**
```
Accuracy : 0.7957
  negative : f1 = 0.87   positive : f1 = 0.84   neutral : f1 = 0.01
```
**Analysis**: Same overall accuracy as model 5.3, but neutrality degrades even further (f1=0.01). Freezing the Word2Vec embeddings appears to limit adaptation to the nuance of neutral reviews.

---

### Model Comparison Summary

| Model | Accuracy | Neutral F1 | Neg F1 | Pos F1 |
|---|---|---|---|---|
| TF-IDF + LR (baseline) | 0.7525 | **0.32** | 0.86 | 0.81 |
| BERT Zero-Shot | 0.7800 | 0.24 | **0.88** | 0.84 |
| Keras Basic Embedding | 0.7957 | 0.09 | 0.86 | 0.85 |
| Keras Word2Vec Embedding | **0.7957** | 0.01 | 0.87 | **0.85** |

**General conclusion**: The `neutral` class is the common weak point across all models. The TF-IDF baseline offers the best trade-off on the neutral class thanks to the `class_weight='balanced'` parameter. Deep learning models outperform the baseline in overall accuracy but sacrifice the neutral class.

---

## 6. Error Analysis (`10_error_analysis.py`)

**Data**: 1,193 errors out of 4,821 test examples (75.3% accuracy).

### True Positives predicted as Negative (63 cases)
Characteristic example:
> *"je suis satisfait du service. j'ai pu trouver une assurance qui rentre dans mes moyens financiers en tout risque. etant jeune conducteur il est difficile de s'assurer sans que cela nous coûte trop cher."*

This text starts positively but includes "coûte trop cher" → the model captured the negative pricing words, even though the overall intent is positive. **Cause: lexical ambivalence.**

### True Negatives predicted as Positive (40 cases)
Example:
> *"suis satisfait du service et les prix attractifs et espérons qu en cas de problème je serais aussi satisfait"*

This text expresses a conditional reservation ("espérons que...") that the model failed to capture. **Cause: understanding of conditionals is impossible with TF-IDF.**

**Summary**: The main sources of error are implicit negation, conditional phrasing, and mixed-sentiment texts (factually positive but with a negative nuance or vice versa).

---

## 7. Category / Topic Detection (`11_category_stars_prediction.py`)

### Method: Zero-Shot Classification (BaptisteDoyen/camembert-base-xnli)
Based on CamemBERT + NLI (Natural Language Inference), without fine-tuning.

**Categories**: `pricing`, `customer service`, `coverage / guarantees`, `reimbursement / claims`, `cancellation`

**Results** (20 samples): Available in `data/processed/sample_categories.csv` and visualized in the "Category Detection" tab of the Streamlit app.

**Advantage**: No manual labeling required. Easily extensible to new categories without retraining.

---

## 8. Streamlit Application (`src/app.py`)

The application provides **5 tabs**:

| Tab | Description |
|---|---|
| **Sentiment & Explanation** | Real-time prediction + word importance (LR coefficients) + comparison table of all 4 models |
| **Insurer Analysis** | Average rating per insurer, sentiment distribution, NLP summaries, keyword search |
| **Semantic Search** | Word2Vec: similar words + filtering of associated reviews |
| **Category Detection** | Live Zero-Shot + pre-computed data on the sample |
| **Summary & QA (RAG)** | Automatic summary of an insurer's reviews + extractive QA (CamemBERT) |

---

## 9. Conclusions & Perspectives

### Strengths
- Complete, fully reproducible end-to-end pipeline
- Diverse approaches: classical ML, deep learning, zero-shot, generative
- Interactive application demonstrating all features

### Identified Limitations
- The `neutral` class is consistently underperforming → requires oversampling (SMOTE) or a dedicated model
- Spell correction (top 2000 words) is a speed/quality trade-off; full correction would improve results
- The T5 summarization model (`plguillou/t5-base-fr-sum-cnndm`) requires a specific pipeline configuration

### Future Improvements
1. Fine-tuning CamemBERT on the labeled corpus for topic detection
2. Using an LLM (e.g. Mistral, LLaMA) in a RAG setup for higher-quality summaries
3. SMOTE or more aggressive class weighting for the neutral class
4. Semantic search via FAISS for similarity search at scale
