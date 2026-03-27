import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

print("Loading dataset and models for error analysis...")
df = pd.read_csv(os.path.join("data", "processed", "cleaned_dataset.csv"))
df = df.dropna(subset=['sentiment', 'avis_clean'])

X = df['avis_clean'].astype(str)
y = df['sentiment']

# Same split state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open(os.path.join("data", "processed", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join("data", "processed", "baseline_lr_model.pkl"), "rb") as f:
    model = pickle.load(f)

print("Predicting...")
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

out_dir = os.path.join("data", "processed")
out_path = os.path.join(out_dir, "error_analysis.txt")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== Error Analysis ===\n\n")
    
    errors = y_test != y_pred
    
    df_errors = pd.DataFrame({
        'Text': X_test[errors],
        'Actual': y_test[errors],
        'Predicted': y_pred[errors]
    })
    
    f.write(f"Total Errors in Test Set: {len(df_errors)} out of {len(y_test)}\n\n")
    
    pos_neg = df_errors[(df_errors['Actual'] == 'positive') & (df_errors['Predicted'] == 'negative')]
    f.write(f"--- Actual Positive, Predicted Negative ({len(pos_neg)}) ---\n")
    for idx, row in pos_neg.head(5).iterrows():
        f.write(f"Text: {row['Text']}\n\n")
        
    neg_pos = df_errors[(df_errors['Actual'] == 'negative') & (df_errors['Predicted'] == 'positive')]
    f.write(f"--- Actual Negative, Predicted Positive ({len(neg_pos)}) ---\n")
    for idx, row in neg_pos.head(5).iterrows():
        f.write(f"Text: {row['Text']}\n\n")
        
print(f"Error analysis saved to {out_path}")
