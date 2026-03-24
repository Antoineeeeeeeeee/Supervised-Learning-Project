import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Loading dataset...")
try:
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    texts = df['avis_clean'].fillna("").tolist()
except FileNotFoundError:
    df = pd.read_csv("data/raw/full_dataset.csv")
    texts = df['avis'].fillna("").astype(str).str.lower().tolist()

# Tokenization
sentences = [str(text).split() for text in texts]

print("Training Word2Vec model...")
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, workers=4)

out_dir = os.path.join("data", "processed")
os.makedirs(out_dir, exist_ok=True)
model.save(os.path.join(out_dir, "word2vec.model"))

print("\n--- Word Similarities & Distances ---")
if 'prix' in model.wv:
    print("Most similar to 'prix':")
    for w, sim in model.wv.most_similar('prix', topn=5):
        print(f" - {w} (cosine sim = {sim:.4f})")
    
    vec1 = model.wv['prix']
    target = 'tarif' if 'tarif' in model.wv else ('cher' if 'cher' in model.wv else None)
    if target:
        vec2 = model.wv[target]
        dist = np.linalg.norm(vec1 - vec2)
        print(f"\nEuclidean distance between 'prix' and '{target}': {dist:.4f}")

print("\nPreparing Matplotlib visualization (PCA)...")
words = list(model.wv.index_to_key)[:200]
vecs = np.array([model.wv[w] for w in words])
pca = PCA(n_components=2)
vecs_2d = pca.fit_transform(vecs)

plt.figure(figsize=(12,12))
plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], alpha=0.5)
for i, w in enumerate(words):
    plt.annotate(w, (vecs_2d[i, 0], vecs_2d[i, 1]), fontsize=8)
plt.title("Word2Vec PCA (First 200 words)")
plt.savefig(os.path.join(out_dir, "word2vec_pca.png"))
print("Saved Matplotlib PCA plot to data/processed/word2vec_pca.png")

print("\nExporting to TensorBoard Projector TSV format...")
out_vecs = os.path.join(out_dir, "tensors.tsv")
out_meta = os.path.join(out_dir, "metadata.tsv")
with open(out_vecs, "w", encoding="utf-8") as fv, open(out_meta, "w", encoding="utf-8") as fm:
    for w in list(model.wv.index_to_key)[:5000]: # Export top 5000 words
        fm.write(w + "\n")
        fv.write("\t".join([str(x) for x in model.wv[w]]) + "\n")
        
print("TensorBoard export complete. Done.")
