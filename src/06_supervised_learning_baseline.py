import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")
df = pd.read_csv(os.path.join("data", "processed", "cleaned_dataset.csv"))

# Drop rows missing sentiment or text
df = df.dropna(subset=['sentiment', 'avis_clean'])

X = df['avis_clean'].astype(str)
y = df['sentiment']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

print("Evaluating...")
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
report = classification_report(y_test, y_pred)
print(report)

print("Saving model and vectorizer...")
with open(os.path.join("data", "processed", "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)
with open(os.path.join("data", "processed", "baseline_lr_model.pkl"), "wb") as f:
    pickle.dump(model, f)
    
# Save metrics for comparison
with open(os.path.join("data", "processed", "model_comparison.txt"), "w", encoding='utf-8') as f:
    f.write("Baseline TF-IDF + Logistic Regression:\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(report + "\n")

print("Done.")
