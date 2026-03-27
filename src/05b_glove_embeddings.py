import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

print("Loading GloVe embeddings via gensim downloader (this may take a moment if not cached)...")
try:
    glove_model = api.load("glove-twitter-25")
except Exception as e:
    print(f"Failed to load GloVe model from gensim api: {e}")
    print("Ensure internet connection or use a local file.")
    exit(1)

out_dir = os.path.join("data", "processed")
os.makedirs(out_dir, exist_ok=True)

print("\n--- Word Similarities & Distances with GloVe ---")
# Since GloVe via this API is typically English, we will test english terms, plus common french if possible
for w in ['price', 'insurance', 'car', 'health', 'prix', 'assurance']:
    if w in glove_model:
        print(f"\nMost similar to '{w}':")
        for sim_w, sim in glove_model.most_similar(w, topn=5):
            print(f" - {sim_w} (cosine sim = {sim:.4f})")
    
if 'price' in glove_model and 'cost' in glove_model:
    vec1 = glove_model['price']
    vec2 = glove_model['cost']
    dist = np.linalg.norm(vec1 - vec2)
    print(f"\nEuclidean distance between 'price' and 'cost': {dist:.4f}")

print("\nPreparing Matplotlib visualization (PCA) for GloVe...")
# Select 200 common words from the vocabulary
words = list(glove_model.index_to_key)[100:300] # Skip top 100 which are often stop words/punctuation
vecs = np.array([glove_model[w] for w in words])
pca = PCA(n_components=2)
vecs_2d = pca.fit_transform(vecs)

plt.figure(figsize=(12,12))
plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], alpha=0.5)
for i, w in enumerate(words):
    plt.annotate(w, (vecs_2d[i, 0], vecs_2d[i, 1]), fontsize=8)
plt.title("GloVe PCA (Words 100-300)")
plt.savefig(os.path.join(out_dir, "glove_pca.png"))
print("Saved Matplotlib PCA plot to data/processed/glove_pca.png")

print("\nExporting to TensorBoard Projector TSV format for GloVe...")
out_vecs = os.path.join(out_dir, "glove_tensors.tsv")
out_meta = os.path.join(out_dir, "glove_metadata.tsv")
with open(out_vecs, "w", encoding="utf-8") as fv, open(out_meta, "w", encoding="utf-8") as fm:
    for w in list(glove_model.index_to_key)[:5000]: # Export top 5000 words
        # Remove any tabs/newlines from words just in case
        clean_w = str(w).replace('\t', '').replace('\n', '')
        fm.write(clean_w + "\n")
        fv.write("\t".join([str(x) for x in glove_model[w]]) + "\n")
        
print("GloVe export complete. Done.")
