import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
try:
    from transformers import pipeline
except ImportError:
    print("Transformers not installed. Please install transformers and torch.")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(os.path.join("data", "processed", "cleaned_dataset.csv"))
df = df.dropna(subset=['sentiment', 'avis_clean'])

X = df['avis_clean'].astype(str)
y = df['sentiment']

# Same split as baseline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sub-sample for speed on CPU
SAMPLE_SIZE = 200
X_test_sample = X_test.iloc[:SAMPLE_SIZE].tolist()
y_test_sample = y_test.iloc[:SAMPLE_SIZE].tolist()

print(f"Loading pre-trained transformer model (Evaluating on {SAMPLE_SIZE} samples)...")
try:
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
except Exception as e:
    print("Failed to load pipeline. Ensure torch or tensorflow is installed.", e)
    exit(1)

print("Predicting...")
preds = classifier(X_test_sample, truncation=True, max_length=512)

# The model outputs labels like '1 star', '2 stars', etc.
def map_stars_to_sentiment(label):
    star = int(label.split()[0])
    if star >= 4:
        return 'positive'
    elif star <= 2:
        return 'negative'
    else:
        return 'neutral'

y_pred_transformer = [map_stars_to_sentiment(p['label']) for p in preds]

acc = accuracy_score(y_test_sample, y_pred_transformer)
print(f"Accuracy of Transformer: {acc:.4f}")
report = classification_report(y_test_sample, y_pred_transformer)
print(report)

# Append to model comparison
with open(os.path.join("data", "processed", "model_comparison.txt"), "a", encoding='utf-8') as f:
    f.write("\nTransformer (nlptown/bert-base-multilingual-uncased-sentiment) - Zero-shot:\n")
    f.write(f"Accuracy (on {SAMPLE_SIZE} sample): {acc:.4f}\n")
    f.write(report + "\n")

print("Done.")
