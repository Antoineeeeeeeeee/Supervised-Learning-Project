import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

st.set_page_config(page_title="Assurance Reviews NLP", layout="wide")

st.title("Assurance Reviews NLP Application 🚀")

@st.cache_resource
def load_models():
    vec_path = os.path.join("data", "processed", "tfidf_vectorizer.pkl")
    mod_path = os.path.join("data", "processed", "baseline_lr_model.pkl")
    try:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(mod_path, "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        return None, None

vectorizer, model = load_models()

@st.cache_data
def load_data():
    data_path = os.path.join("data", "processed", "cleaned_dataset.csv")
    try:
        df = pd.read_csv(data_path)
        return df
    except:
        return pd.DataFrame()

df = load_data()

tab1, tab2 = st.tabs(["🔮 Prédiction de Sentiment", "📊 Analyse & Recherche d'Assureurs"])

with tab1:
    st.header("Tester l'analyse de sentiment")
    user_input = st.text_area("Entrez un avis sur une assurance :", height=150)
    
    if st.button("Analyser"):
        if user_input and vectorizer and model:
            vec_input = vectorizer.transform([user_input])
            pred = model.predict(vec_input)[0]
            probs = model.predict_proba(vec_input)[0]
            
            st.subheader(f"Sentiment Prédit : **{pred.upper()}**")
            
            st.write("### Explication de la prédiction")
            feature_names = vectorizer.get_feature_names_out()
            class_index = list(model.classes_).index(pred)
            coefs = model.coef_[class_index]
            
            words_in_text = user_input.lower().split()
            word_impacts = []
            for w in set(words_in_text):
                if w in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[w]
                    impact = coefs[idx] * vec_input[0, idx]
                    word_impacts.append((w, impact))
            
            word_impacts.sort(key=lambda x: x[1], reverse=True)
            
            st.write("Mots ayant le plus contribué à cette prédiction :")
            for w, imp in word_impacts[:5]:
                if imp > 0:
                    st.write(f"- **{w}** (Impact: +{imp:.4f})")
            
            st.progress(float(np.max(probs)))
        else:
            st.warning("Veuillez entrer du texte ou vérifier que le modèle est bien généré.")

with tab2:
    st.header("Exploration et Recherche")
    if not df.empty:
        assureurs = df['assureur'].dropna().unique().tolist()
        selected_assureurs = st.multiselect("Filtrer par assureur :", assureurs)
        
        filtered_df = df[df['assureur'].isin(selected_assureurs)] if selected_assureurs else df
        
        st.write("### Moyenne des notes par Assureur")
        if 'note' in filtered_df.columns:
            avg_stars = filtered_df.groupby('assureur')['note'].mean().reset_index()
            st.bar_chart(avg_stars.set_index('assureur'))
        
        st.write("### Recherche de Mots-clés (Information Retrieval)")
        keyword = st.text_input("Mots-clés :")
        if keyword:
            res = filtered_df[filtered_df['avis'].astype(str).str.contains(keyword, case=False, na=False)]
            st.dataframe(res[['assureur', 'note', 'avis']].head(50))
    else:
        st.error("Dataset introuvable. Veuillez exécuter les scripts de préparation des données.")
