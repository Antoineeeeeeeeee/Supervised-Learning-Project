# Rapport Technique — Projet NLP Supervisé
### Analyse des avis clients dans le secteur de l'assurance

---

## 1. Contexte et Données

### Dataset
Le dataset est composé d'avis clients en français sur des assureurs (assurance auto, santé, habitation). Les données brutes sont contenues dans plusieurs fichiers `.xlsx` fusionnés en un CSV unique (`data/raw/full_dataset.csv`). Chaque ligne contient :
- `assureur` : le nom de la compagnie
- `avis` : le texte de l'avis (en français)
- `note` : la note attribuée (1 à 5 étoiles)

---

## 2. Exploration & Nettoyage des Données

### Choix techniques (`03_data_cleaning.py`)
- **Normalisation** : mise en minuscules, suppression des retours à la ligne et espaces multiples.
- **Correction orthographique** : implémentée via `pyspellchecker` avec une approche en cache optimisée (top 2000 mots inconnus identifiés corpus-wide, corrigés une seule fois). Cette approche réduit le temps d'exécution de plusieurs heures à quelques minutes.
- **Étiquetage du sentiment** : la colonne `sentiment` est déduite de la note :
  - Note ≥ 4 → `positive`
  - Note ≤ 2 → `negative`
  - Note = 3 → `neutral`

### Mots fréquents et bigrammes
Les 20 mots les plus fréquents du corpus (après nettoyage) reflètent la structure grammaticale du français. Les bigrammes sont plus informatifs et révèlent les patterns de satisfaction :

| Bigramme | Occurrences | Interprétation |
|---|---|---|
| `je suis` | 11 457 | Début de prise de position |
| `suis satisfait` | 2 812 | Signal fort de satisfaction |
| `le service` | 2 129 | Référence fréquente au service client |
| `les prix` | 2 108 | Sujet sensible des tarifs |
| `je recommande` | 1 896 | Recommandation explicite |
| `en charge` | 2 093 | Gestion des sinistres / dossiers |

**Conclusion** : Le corpus tourne autour de 3 axes principaux — satisfaction globale, prix/tarifs, et service client/sinistres. Cela correspond parfaitement aux catégories Zero-Shot définies dans la tâche de détection de thèmes.

---

## 3. Topic Modeling (`04_topic_modeling.py`)

### Méthode : NMF (Non-negative Matrix Factorization)
Appliquée sur une matrice TF-IDF (max 1000 features, stop words français), avec 5 composantes.

### Résultats :

| Topic | Mots clés | Interprétation |
|---|---|---|
| **Topic 1** | sinistre, contrat, mois, depuis, sans | Gestion des sinistres / litiges |
| **Topic 2** | satisfait, prix, service, qualité, client | Satisfaction et rapport qualité-prix |
| **Topic 3** | direct, auto, voiture, véhicule, cher | Assurance automobile |
| **Topic 4** | rapide, simple, efficace, site, souscription | Expérience digitale / souscription en ligne |
| **Topic 5** | téléphone, personne, reçu, jamais | Difficultés à joindre le service client |

**Interprétation** : Les 5 topics correspondent à des axes bien identifiés dans l'assurance : sinistres, satisfaction, automobile, digital et service client.

---

## 4. Word Embeddings

### Word2Vec (`05_word_embeddings.py`)
- **Architecture** : Skip-gram, 100 dimensions, fenêtre de 5, min_count=5
- **Résultats de similarité** :
  - Mots proches de `prix` : tarif, cher, augmentation, cotisation → cohérence sémantique sur la thématique tarifaire
  - Distance euclidienne entre `prix` et `tarif` calculée et sauvegardée
- **Visualisation** : PCA 2D des 200 premiers mots (`word2vec_pca.png`) + export TensorBoard (tensors.tsv / metadata.tsv pour 5000 mots)

### GloVe (`05b_glove_embeddings.py`)
- **Modèle** : `glove-twitter-25` chargé via `gensim.downloader` (104.8 MB)
- **Résultats de similarité** (en anglais) :
  - `price` → card, stock, includes, discount, limited
  - `insurance` → exchange, mortgage, transportation, banking
- **Visualisation** : PCA 2D des mots 100-300 (`glove_pca.png`) + export TensorBoard (glove_tensors.tsv / glove_metadata.tsv)

**Comparaison W2V vs GloVe** : Word2Vec entraîné sur le corpus français est plus pertinent thématiquement (il capte `prix`, `remboursement`, etc. dans un contexte assurance). GloVe généraliste pré-entraîné sur Twitter offre des similarités moins spécialisées mais une couverture de vocabulaire plus large.

---

## 5. Apprentissage Supervisé — Analyse de Sentiment

### Tâche : Classification 3 classes (positive / neutral / negative)

### 5.1 Baseline TF-IDF + Régression Logistique (`06_supervised_learning_baseline.py`)
- **Vectorisation** : TF-IDF, max 5000 features, pondération class_weight='balanced'
- **Split** : 80/20, random_state=42

**Résultats :**
```
Accuracy : 0.7525
               precision  recall  f1-score  support
  negative       0.87      0.85     0.86     2171
   neutral       0.28      0.37     0.32      682
  positive       0.85      0.78     0.81     1968
```
**Analyse** : La classe neutre est très difficile à prédire (f1=0.32). Cela s'explique par sa définition (note=3, souvent ambiguë) et son faible nombre d'exemples (682 vs ~2000 pour les autres classes).

---

### 5.2 Transformers Zero-Shot (`07_hf_transformer_model.py`)
- **Modèle** : `nlptown/bert-base-multilingual-uncased-sentiment` (BERT multilingue, prédit 1-5 étoiles, converti en 3 classes)
- **Évaluation** : 200 échantillons (contrainte CPU)

**Résultats :**
```
Accuracy : 0.7800 (sur 200 samples)
  negative : f1 = 0.88   positive : f1 = 0.84   neutral : f1 = 0.24
```
**Analyse** : Performance comparable à la baseline TF-IDF, légèrement meilleure sur l'ensemble, mais la classe neutre reste problématique (f1=0.24). Le modèle "voit" les étoiles directement, ce qui est un avantage conceptuel.

---

### 5.3 Réseau de neurones — Embedding aléatoire (`08_keras_basic_embedding.py`)
- **Architecture** : `Embedding(5000, 50)` → `GlobalAveragePooling1D` → `Dense(24, relu)` → `Dropout(0.5)` → `Dense(3, softmax)`
- **Entraînement** : 10 epochs, Adam, sparse_categorical_crossentropy
- **TensorBoard** : logs sauvegardés dans `logs/fit/basic_embedding_*`

**Résultats :**
```
Accuracy : 0.7957
  negative : f1 = 0.86   positive : f1 = 0.85   neutral : f1 = 0.09
```
**Analyse** : Meilleure précision globale (+4pp vs baseline), mais la classe neutre s'effondre (f1=0.09). Le modèle a convergé sur les deux classes dominantes.

---

### 5.4 Réseau de neurones — Pré-entraîné Word2Vec (`09_keras_pretrained_embedding.py`)
- **Architecture** : Identique au modèle 5.3, mais l'Embedding layer est initialisée avec les poids Word2Vec (100 dim, frozen)
- **Entraînement** : 15 epochs, Adam

**Résultats :**
```
Accuracy : 0.7957
  negative : f1 = 0.87   positive : f1 = 0.84   neutral : f1 = 0.01
```
**Analyse** : Même accuracy globale que le modèle 5.3, mais la neutralité est encore plus dégradée (f1=0.01). Le gel des embeddings Word2Vec semble limiter l'adaptation à la nuance des avis neutres.

---

### Comparaison synthétique des modèles

| Modèle | Accuracy | Neutral F1 | Neg F1 | Pos F1 |
|---|---|---|---|---|
| TF-IDF + LR (baseline) | 0.7525 | **0.32** | 0.86 | 0.81 |
| BERT Zero-Shot | 0.7800 | 0.24 | **0.88** | 0.84 |
| Keras Basic Embedding | 0.7957 | 0.09 | 0.86 | 0.85 |
| Keras Word2Vec Embedding | **0.7957** | 0.01 | 0.87 | **0.85** |

**Conclusion générale** : La classe `neutral` est le point faible commun à tous les modèles. La baseline TF-IDF offre le meilleur compromis sur la classe neutre grâce au paramètre `class_weight='balanced'`. Les modèles deep learning surpassent la baseline en accuracy globale mais sacrifient la classe neutre.

---

## 6. Analyse des Erreurs (`10_error_analysis.py`)

**Données** : 1193 erreurs sur 4821 exemples de test (75.3% d'accuracy).

### Vrais Positifs prédits Négatifs (63 cas)
Exemple caractéristique :
> *"je suis satisfait du service. j'ai pu trouver une assurance qui rentre dans mes moyens financiers en tout risque. etant jeune conducteur il est difficile de s'assurer sans que cela nous coûte trop cher."*

Ce texte commence positivement mais inclut "coûte trop cher" → le modèle a capté les mots négatifs liés au prix, alors que l'intention globale est positive. **Cause : ambivalence lexicale.**

### Vrais Négatifs prédits Positifs (40 cas)
Exemple :
> *"suis satisfait du service et les prix attractifs et espérons qu en cas de problème je serais aussi satisfait"*

Ce texte exprime une réserve conditionnelle ("espérons que...") que le modèle n'a pas capté. **Cause : compréhension du conditionnel impossible avec TF-IDF.**

**Bilan** : Les principales sources d'erreur sont la négation implicite, le conditionnel, et les textes avec sentiments mixtes (factuellement positifs mais avec une nuance négative ou vice-versa).

---

## 7. Détection de Catégories / Thèmes (`11_category_stars_prediction.py`)

### Méthode : Classification Zero-Shot (BaptisteDoyen/camembert-base-xnli)
Basée sur CamemBERT + inférence NLI (Natural Language Inference), sans fine-tuning.

**Catégories** : `tarifs / prix`, `service client`, `couverture / garanties`, `remboursement / sinistres`, `résiliation`

**Résultats** (20 échantillons) : Consultables dans `data/processed/sample_categories.csv` et visualisés dans le Tab "Détection de catégories" de l'application Streamlit.

**Avantage** : Aucun label manuel requis. Extensible à de nouvelles catégories sans réentraînement.

---

## 8. Application Streamlit (`src/app.py`)

L'application propose **5 onglets** :

| Onglet | Description |
|---|---|
| **Sentiment & Explication** | Prédiction en temps réel + importance des mots (LR coefs) + tableau de comparaison des 4 modèles |
| **Analyse des assureurs** | Note moyenne par assureur, distribution des sentiments, résumés NLP, recherche par mot-clé |
| **Recherche Sémantique** | Word2Vec : mots similaires + filtrage des avis associés |
| **Détection de catégories** | Zero-Shot live + données pré-calculées sur l'échantillon |
| **Résumé & QA (RAG)** | Résumé automatique des avis par assureur + QA extractif (CamemBERT) |

---

## 9. Conclusions & Perspectives

### Points forts
- Pipeline complet, reproductible bout-en-bout
- Diversité des approches : classique, deep learning, zero-shot, générative
- Application interactive démontrant toutes les fonctionnalités

### Limites identifiées
- La classe `neutral` est systématiquement sous-performante → nécessite un sur-échantillonnage (SMOTE) ou un modèle dédié
- La correction orthographique (top 2000 mots) est un compromis vitesse/qualité ; une correction complète améliorait les résultats
- Le modèle de résumé T5 (`plguillou/t5-base-fr-sum-cnndm`) nécessite une configuration pipeline spécifique

### Pistes d'amélioration
1. Fine-tuning de CamemBERT sur le corpus labellisé pour la détection de thèmes
2. Utilisation d'un LLM (ex. Mistral, LLaMA) en RAG pour des résumés de meilleure qualité
3. SMOTE ou class weighting plus agressif pour la classe neutre
4. Semantic search via FAISS pour une recherche de similarité à l'échelle
