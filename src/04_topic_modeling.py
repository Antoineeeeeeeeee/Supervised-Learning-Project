import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import os

print("Loading dataset for Topic Modeling...")
try:
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    texts = df['avis_clean'].fillna("").tolist()
except FileNotFoundError:
    print("Cleaned dataset not found. Using raw dataset with basic normalization as fallback...")
    df = pd.read_csv("data/raw/full_dataset.csv")
    texts = df['avis'].fillna("").astype(str).str.lower().tolist()

print("Vectorizing...")
french_stop_words = ["le", "la", "les", "un", "une", "des", "du", "de", "et", "en", "à", "au", "aux", "pour", "dans", "sur", "qui", "que", "quoi", "dont", "où", "mais", "ou", "donc", "or", "ni", "car", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses", "ce", "cet", "cette", "ces", "pas", "plus", "très", "bien", "fait", "tout", "tous", "comme", "avec", "est", "sont", "a", "ont", "suis", "il", "y", "ne", "se", "qu", "c'est", "j'ai", "m'a", "l'a", "d'un", "qu'il"]

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, max_features=1000, stop_words=french_stop_words)
tfidf = tfidf_vectorizer.fit_transform(texts)

print("Fitting NMF...")
n_components = 5
nmf = NMF(n_components=n_components, random_state=1, init='nndsvda', solver='mu', max_iter=200).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

topics = {}
for topic_idx, topic in enumerate(nmf.components_):
    top_features_ind = topic.argsort()[:-10 - 1:-1]
    top_features = [tfidf_feature_names[i] for i in top_features_ind]
    topics[f"Topic {topic_idx+1}"] = top_features
    print(f"Topic {topic_idx+1}: {', '.join(top_features)}")

out_path = os.path.join("data", "processed", "topics.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for topic, words in topics.items():
        f.write(f"{topic}: {', '.join(words)}\n")
print(f"Saved topics to {out_path}")
