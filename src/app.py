import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# Try importing extra NLP libraries for new features
try:
    from gensim.models import Word2Vec
    HAS_W2V = True
except ImportError:
    HAS_W2V = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

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
    except Exception:
        return None, None

@st.cache_resource
def load_w2v():
    if not HAS_W2V: return None
    path = os.path.join("data", "processed", "word2vec.model")
    if os.path.exists(path):
        return Word2Vec.load(path)
    return None

@st.cache_resource
def load_pipelines():
    if not HAS_TRANSFORMERS: return None, None
    try:
        qa_pipe = pipeline("question-answering", model="etalab-ia/camembert-base-squadFR-fquad-piaf", revision="main") 
        sum_pipe = pipeline("summarization", model="plguillou/t5-base-fr-sum-cnndm")
        return qa_pipe, sum_pipe
    except:
        try:
            qa_pipe = pipeline("question-answering")
            sum_pipe = pipeline("summarization")
            return qa_pipe, sum_pipe
        except:
            return None, None

vectorizer, model = load_models()
w2v_model = load_w2v()
qa_pipe, sum_pipe = load_pipelines()

@st.cache_data
def load_data():
    data_path = os.path.join("data", "processed", "cleaned_dataset.csv")
    try:
        df = pd.read_csv(data_path)
        return df
    except:
        return pd.DataFrame()

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Sentiment", "📊 Analyse", "🧠 Recherche Sémantique", "🤖 RAG & Résumés"])

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
            st.warning("Erreur : Text vide ou modèle non chargé.")

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
        
        st.write("### Recherche par mot-clé exact (Information Retrieval)")
        keyword = st.text_input("Mot-clé exact :")
        if keyword:
            res = filtered_df[filtered_df['avis'].astype(str).str.contains(keyword, case=False, na=False)]
            st.dataframe(res[['assureur', 'note', 'avis']].head(50))
    else:
        st.error("Dataset introuvable.")

with tab3:
    st.header("Recherche Sémantique (Word2Vec)")
    if w2v_model and not df.empty:
        query = st.text_input("Recherchez un concept (ex: 'prix', 'remboursement') :")
        if st.button("Chercher sémantiquement") and query:
            if query in w2v_model.wv:
                sims = w2v_model.wv.most_similar(query, topn=3)
                sim_words = [query] + [w for w, _ in sims]
                st.write(f"Mots sémantiquement liés trouvés : **{', '.join(sim_words)}**")
                
                # Filtrer les avis contenant l'un des mots
                pattern = '|'.join(sim_words)
                res = df[df['avis_clean'].astype(str).str.contains(pattern, case=False, na=False)]
                st.write(f"{len(res)} avis trouvés :")
                st.dataframe(res[['assureur', 'note', 'avis']].head(20))
            else:
                st.warning("Mot inconnu dans le vocabulaire Word2Vec.")
    else:
        st.warning("Modèle Word2Vec non chargé ou introuvable.")

with tab4:
    st.header("Fonctions Avancées : Résumé & QA (RAG)")
    if HAS_TRANSFORMERS and not df.empty:
        assureur_rag = st.selectbox("Sélectionnez un assureur pour analyser les avis :", df['assureur'].dropna().unique().tolist())
        target_df = df[df['assureur'] == assureur_rag]
        
        st.subheader("📝 Résumé des avis")
        if st.button("Générer un résumé des 10 derniers avis"):
            texts = target_df['avis'].dropna().head(10).tolist()
            combined = " ".join([str(t) for t in texts])[:1000] # truncate
            if sum_pipe:
                with st.spinner("Génération en cours..."):
                    try:
                        summary = sum_pipe(combined, max_length=150, min_length=30, do_sample=False)
                        st.success(summary[0]['summary_text'])
                    except Exception as e:
                        st.error(f"Erreur lors de la génération: {e}")
            else:
                st.error("Pipeline summarization non disponible.")
                
        st.subheader("❓ Poser une question sur cet assureur (QA)")
        question = st.text_input("Votre question (ex: 'Quels sont les problèmes ?') :")
        if st.button("Demander"):
            if qa_pipe and question:
                texts = target_df['avis'].dropna().head(50).tolist() # contexte
                context = " ".join([str(t) for t in texts])[:2000]
                with st.spinner("Recherche de la réponse..."):
                    try:
                        ans = qa_pipe(question=question, context=context)
                        st.success(f"Réponse: {ans['answer']} (Confiance: {ans['score']:.2f})")
                    except Exception as e:
                        st.error(f"Erreur QA: {e}")
            else:
                st.error("Pipeline QA non disponible ou question vide.")
    else:
        st.warning("Transformers non installé ou dataset vide.")
