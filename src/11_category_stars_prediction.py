import os
import pandas as pd
try:
    from transformers import pipeline
except ImportError:
    print("Transformers not installed. Please install it to run zero-shot classification.")
    exit(1)

print("Loading dataset...")
df = pd.read_csv(os.path.join("data", "processed", "cleaned_dataset.csv"))
df = df.dropna(subset=['sentiment', 'avis_clean'])

sample_size = 20
df_sample = df.sample(n=sample_size, random_state=42).copy()

print("Loading Zero-Shot Classification Pipeline...")
try:
    classifier = pipeline("zero-shot-classification", model="BaptisteDoyen/camembert-base-xnli") 
except Exception as e:
    print(f"Failed to load French pipeline, using default: {e}")
    classifier = pipeline("zero-shot-classification")

candidate_labels = ["tarifs", "service client", "couverture", "remboursement"]

print(f"Predicting categories for {sample_size} samples...")
predictions = []
for text in df_sample['avis_clean']:
    res = classifier(str(text)[:512], candidate_labels)
    best_label = res['labels'][0]
    predictions.append(best_label)

df_sample['predicted_category'] = predictions

out_dir = os.path.join("data", "processed")
out_path = os.path.join(out_dir, "sample_categories.csv")
df_sample[['avis_clean', 'predicted_category']].to_csv(out_path, index=False)

print("\nSample Predictions:")
print(df_sample[['avis_clean', 'predicted_category']].head(10))
print(f"\nSaved sample predictions to {out_path}")
