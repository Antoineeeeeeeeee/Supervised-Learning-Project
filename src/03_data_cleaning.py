import pandas as pd
import re
import os
import time
from collections import Counter

try:
    from spellchecker import SpellChecker
    HAS_SPELL = True
except ImportError:
    HAS_SPELL = False
    print("pyspellchecker not installed.")

print("Loading dataset...")
df = pd.read_csv("data/raw/full_dataset.csv")

# 1. Map note to sentiment
def map_sentiment(note):
    if pd.isna(note):
        return None
    if note >= 4:
        return 'positive'
    elif note <= 2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['note'].apply(map_sentiment)

# 2. Text Normalization
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Normalizing texts...")
df['avis_norm'] = df['avis'].apply(normalize_text)

# 3. Fast Spelling Correction
if HAS_SPELL:
    print("Applying spell correction (fast cached approach)...")
    start_time = time.time()
    spell = SpellChecker(language='fr')
    
    # Extract all words
    all_words = []
    for text in df['avis_norm']:
        all_words.extend(re.findall(r'\b[a-zàâäéèêëïîôöùûüÿç]+\b', text))
        
    word_counts = Counter(all_words)
    unique_words = set(word_counts.keys())
    
    print(f"Total words: {len(all_words)}, Unique: {len(unique_words)}")
    
    unknown_words = spell.unknown(unique_words)
    print(f"Found {len(unknown_words)} unknown words.")
    
    unknown_word_counts = Counter({w: c for w, c in word_counts.items() if w in unknown_words})
    
    # Correct top 2000 most common spelling mistakes to save time
    correction_cache = {}
    top_unknowns = unknown_word_counts.most_common(2000)
    for w, _ in top_unknowns:
        # Some French words might not be corrected properly, but it's okay for an automated tool
        corrected = spell.correction(w)
        if corrected:
            correction_cache[w] = corrected
    
    print(f"Built correction cache in {time.time() - start_time:.2f} seconds.")

    def correct_text(text):
        if not text:
            return ""
        # Replace only words in the cache
        def replace_match(match):
            word = match.group(0)
            return correction_cache.get(word, word)
        return re.sub(r'\b[a-zàâäéèêëïîôöùûüÿç]+\b', replace_match, text)

    df['avis_clean'] = df['avis_norm'].apply(correct_text)
else:
    df['avis_clean'] = df['avis_norm']

out_path = os.path.join("data", "processed", "cleaned_dataset.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Cleaned dataset saved to {out_path} with {len(df)} rows.")
