def load_data(filepath):
    import pandas as pd
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    else:
        raise ValueError('Format non supporté')

def load_multiple_excels(folder_path):
    import os
    import pandas as pd
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    dfs = [pd.read_excel(os.path.join(folder_path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def clean_text(text):
    # Nettoyage du texte, correction orthographique
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # TODO: correction orthographique
    return text

def extract_ngrams(text, n=2):
    # Extraction des n-grams
    from nltk import ngrams
    tokens = text.split()
    return list(ngrams(tokens, n))

def save_cleaned_data(data, filepath):
    # Sauvegarder les données nettoyées
    data.to_csv(filepath, index=False)

def preview_data(data, n=5):
    # Aperçu des premières lignes
    return data.head(n)

def frequent_words(data, text_column, top_n=20):
    # Afficher les mots les plus fréquents
    from collections import Counter
    words = []
    for text in data[text_column]:
        words.extend(clean_text(str(text)).split())
    return Counter(words).most_common(top_n)
