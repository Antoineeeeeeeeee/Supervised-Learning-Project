import os
import pandas as pd
import numpy as np
import datetime

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    print("TensorFlow not installed. Please install it to run Keras models.")
    exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec

print("Loading dataset...")
df = pd.read_csv(os.path.join("data", "processed", "cleaned_dataset.csv"))
df = df.dropna(subset=['sentiment', 'avis_clean'])

X = df['avis_clean'].astype(str).tolist()
y_str = df['sentiment'].tolist()

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
y = np.array([label_mapping[label] for label in y_str])

print("Tokenizing and Padding...")
vocab_size = 5000
max_length = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

print("Loading pre-trained Word2Vec model...")
w2v_path = os.path.join("data", "processed", "word2vec.model")
if not os.path.exists(w2v_path):
    print(f"Word2Vec model not found at {w2v_path}. Please run 05_word_embeddings.py first.")
    exit(1)

w2v_model = Word2Vec.load(w2v_path)
embedding_dim = w2v_model.vector_size

print("Creating Embedding Matrix...")
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= vocab_size:
        break
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

print("Building Keras Pre-trained Embedding Model...")
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False, name="pretrained_embedding"),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("Training...")
log_dir = os.path.join("logs", "fit", "pretrained_embedding_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, embeddings_freq=1)

epochs = 15
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[tensorboard_callback], verbose=1)

print("Evaluating...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
target_names = ['negative', 'neutral', 'positive']
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

out_dir = os.path.join("data", "processed")
model.save(os.path.join(out_dir, "pretrained_embedding_model.keras"))

with open(os.path.join(out_dir, "model_comparison.txt"), "a", encoding='utf-8') as f:
    f.write("\nKeras Pre-Trained (Word2Vec) Embedding Model:\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(report + "\n")

print(f"Done. TensorBoard logs saved to {log_dir}")
