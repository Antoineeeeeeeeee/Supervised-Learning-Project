import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ── Optional NLP imports ──────────────────────────────────────
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
st.title("🏥 Assurance Reviews — NLP Application")

# ── Loaders ───────────────────────────────────────────────────
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
    if not HAS_W2V:
        return None
    path = os.path.join("data", "processed", "word2vec.model")
    if os.path.exists(path):
        return Word2Vec.load(path)
    return None

@st.cache_resource
def load_zeroshot():
    if not HAS_TRANSFORMERS:
        return None
    try:
        return pipeline("zero-shot-classification", model="BaptisteDoyen/camembert-base-xnli")
    except Exception:
        try:
            return pipeline("zero-shot-classification")
        except Exception:
            return None

@st.cache_resource
def load_pipelines():
    if not HAS_TRANSFORMERS:
        return None, None
    try:
        qa_pipe = pipeline("question-answering", model="etalab-ia/camembert-base-squadFR-fquad-piaf", revision="main")
        # T5 French summarization requires text2text-generation task
        sum_pipe = pipeline("text2text-generation", model="plguillou/t5-base-fr-sum-cnndm")
        return qa_pipe, sum_pipe
    except Exception:
        try:
            return pipeline("question-answering"), pipeline("text2text-generation")
        except Exception:
            return None, None

@st.cache_data
def load_data():
    path = os.path.join("data", "processed", "cleaned_dataset.csv")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_model_comparison():
    path = os.path.join("data", "processed", "model_comparison.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

@st.cache_data
def load_insurer_summaries():
    path = os.path.join("data", "processed", "insurer_summaries.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

vectorizer, model = load_models()
w2v_model = load_w2v()
zeroshot_pipe = load_zeroshot()
qa_pipe, sum_pipe = load_pipelines()
df = load_data()
model_comparison_txt = load_model_comparison()
df_summaries = load_insurer_summaries()

CANDIDATE_LABELS = ["tarifs / prix", "service client", "couverture / garanties", "remboursement / sinistres", "résiliation"]

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Sentiment & Explication",
    "📊 Analyse des assureurs",
    "🧠 Recherche Sémantique",
    "🏷️ Détection de catégories",
    "🤖 Résumé & QA (RAG)"
])

# ── TAB 1 : Sentiment Prediction + Explanation ────────────────
with tab1:
    st.header("Analyse de sentiment d'un avis")
    st.caption("Tapez un avis en français. Le modèle TF-IDF + Logistic Regression prédit le sentiment et explique sa décision.")
    user_input = st.text_area("Votre avis :", height=150,
                              placeholder="Ex: L'assurance est trop chère et le service client est horrible.")
    if st.button("Analyser le sentiment", key="btn_sentiment"):
        if user_input and vectorizer and model:
            vec_input = vectorizer.transform([user_input])
            pred = model.predict(vec_input)[0]
            probs = model.predict_proba(vec_input)[0]
            class_labels = list(model.classes_)

            emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(pred, "⚪")
            st.subheader(f"{emoji} Sentiment prédit : **{pred.upper()}**")

            prob_df = pd.DataFrame({"Classe": class_labels, "Probabilité": probs * 100})\
                        .sort_values("Probabilité", ascending=False)
            st.bar_chart(prob_df.set_index("Classe"))

            st.write("**Mots les plus influents :**")
            class_index = class_labels.index(pred)
            coefs = model.coef_[class_index]
            word_impacts = []
            for w in set(user_input.lower().split()):
                if w in vectorizer.vocabulary_:
                    idx = vectorizer.vocabulary_[w]
                    impact = float(coefs[idx] * vec_input[0, idx])
                    word_impacts.append({"Mot": w, "Impact": round(impact, 4)})
            word_impacts.sort(key=lambda x: x["Impact"], reverse=True)
            if word_impacts:
                st.dataframe(pd.DataFrame(word_impacts[:10]), use_container_width=True)
        else:
            st.warning("Entrez du texte et assurez-vous que les modèles sont chargés.")

    st.divider()
    st.subheader("📈 Comparaison des modèles entraînés")
    if model_comparison_txt:
        st.code(model_comparison_txt, language=None)
    else:
        st.info("Lancez les scripts 06 à 09 pour générer `model_comparison.txt`.")

# ── TAB 2 : Insurer Analysis ──────────────────────────────────
with tab2:
    st.header("Exploration des assureurs")
    if not df.empty:
        assureurs = sorted(df['assureur'].dropna().unique().tolist())
        selected = st.multiselect("Filtrer par assureur :", assureurs)
        filtered_df = df[df['assureur'].isin(selected)] if selected else df

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Moyenne des étoiles par assureur**")
            if 'note' in filtered_df.columns:
                avg = filtered_df.groupby('assureur')['note'].mean().sort_values(ascending=False)
                st.bar_chart(avg)
        with col2:
            st.write("**Distribution des sentiments**")
            if 'sentiment' in filtered_df.columns:
                sent_counts = filtered_df['sentiment'].value_counts()
                st.bar_chart(sent_counts)

        st.write("**Résumés NLP par assureur** *(générés par `summary_translation.py`)*")
        if df_summaries is not None:
            selected_assureur_sum = st.selectbox("Voir le résumé de :", assureurs, key="sumselect")
            row = df_summaries[df_summaries['assureur'] == selected_assureur_sum]
            if not row.empty:
                st.info(row.iloc[0]['summary'])
            else:
                st.warning("Pas de résumé disponible pour cet assureur.")
        else:
            st.info("Lancez `python src/summary_translation.py` pour générer les résumés.")

        st.write("**Recherche par mot-clé**")
        keyword = st.text_input("Mot-clé :")
        if keyword:
            res = filtered_df[filtered_df['avis'].astype(str).str.contains(keyword, case=False, na=False)]
            st.write(f"{len(res)} avis trouvés :")
            st.dataframe(res[['assureur', 'note', 'avis']].head(50))
    else:
        st.error("Dataset introuvable. Lancez les scripts de préparation de données.")

# ── TAB 3 : Semantic Search ───────────────────────────────────
with tab3:
    st.header("Recherche Sémantique (Word2Vec)")
    st.caption("Cherchez un concept. Word2Vec trouve les mots associés et filtre les avis les contenant.")
    if w2v_model and not df.empty:
        query = st.text_input("Mot à rechercher sémantiquement :",
                              placeholder="Ex: prix, remboursement, annulation")
        topn = st.slider("Nombre de mots similaires :", 1, 10, 3)
        if st.button("Chercher", key="btn_sem"):
            q = query.strip().lower()
            if q in w2v_model.wv:
                sims = w2v_model.wv.most_similar(q, topn=topn)
                sim_words = [q] + [w for w, _ in sims]

                sim_df = pd.DataFrame(sims, columns=["Mot similaire", "Similarité cosinus"])
                st.write(f"**{topn} mots les plus proches de '{q}' :**")
                st.dataframe(sim_df.style.format({"Similarité cosinus": "{:.4f}"}), use_container_width=True)

                pattern = '|'.join([rf'\b{w}\b' for w in sim_words])
                res = df[df['avis_clean'].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
                st.write(f"**{len(res)} avis trouvés contenant ces mots :**")
                st.dataframe(res[['assureur', 'note', 'avis']].head(30))
            else:
                st.warning(f"Le mot **'{q}'** n'est pas dans le vocabulaire Word2Vec. Essayez un autre mot.")
    elif not HAS_W2V:
        st.error("Gensim non installé.")
    else:
        st.warning("Modèle Word2Vec introuvable. Lancez `python src/05_word_embeddings.py` d'abord.")

# ── TAB 4 : Category Detection (Zero-Shot) ───────────────────
with tab4:
    st.header("Détection de catégories / thèmes (Zero-Shot)")
    st.caption("Classifie instantanément un avis dans une des catégories thématiques de l'assurance.")

    st.write(f"**Catégories disponibles :** {', '.join(CANDIDATE_LABELS)}")

    user_text_cat = st.text_area("Entrez un avis :", height=130,
                                  placeholder="Ex: Mon remboursement a pris 3 mois, c'est inacceptable.",
                                  key="cat_input")
    if st.button("Détecter la catégorie", key="btn_cat"):
        if user_text_cat and zeroshot_pipe:
            with st.spinner("Analyse en cours..."):
                result = zeroshot_pipe(user_text_cat[:512], CANDIDATE_LABELS)
            cat_df = pd.DataFrame({
                "Catégorie": result["labels"],
                "Score (%)": [round(s * 100, 2) for s in result["scores"]]
            })
            st.subheader(f"🏷️ Catégorie principale : **{result['labels'][0]}**")
            st.bar_chart(cat_df.set_index("Catégorie"))
        elif not zeroshot_pipe:
            st.error("Pipeline zero-shot non disponible. Vérifiez l'installation de Transformers et le modèle.")
        else:
            st.warning("Entrez un texte.")

    st.divider()
    st.subheader("📁 Prédictions sur un échantillon existant")
    sample_path = os.path.join("data", "processed", "sample_categories.csv")
    if os.path.exists(sample_path):
        df_cats = pd.read_csv(sample_path)
        st.dataframe(df_cats, use_container_width=True)
        dist = df_cats["predicted_category"].value_counts()
        st.bar_chart(dist)
    else:
        st.info("Lancez `python src/11_category_stars_prediction.py` pour générer les données.")

# ── TAB 5 : RAG / QA / Summarization ─────────────────────────
with tab5:
    st.header("Résumé NLP & Question-Réponse (RAG)")
    if HAS_TRANSFORMERS and not df.empty:
        assureur_rag = st.selectbox(
            "Sélectionnez un assureur :",
            sorted(df['assureur'].dropna().unique().tolist()),
            key="rag_assureur"
        )
        target_df = df[df['assureur'] == assureur_rag]
        st.caption(f"**{len(target_df)} avis** disponibles pour cet assureur.")

        st.subheader("📝 Résumé automatique des avis")
        if st.button("Générer un résumé (10 avis)", key="btn_sum"):
            texts = target_df['avis'].dropna().head(10).tolist()
            combined = " ".join([str(t) for t in texts])[:1200]
            if sum_pipe:
                with st.spinner("Génération en cours..."):
                    try:
                        summary = sum_pipe(combined, max_length=150, min_length=30, do_sample=False)
                        # text2text-generation returns `generated_text`
                        st.success(summary[0]['generated_text'])
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.error("Pipeline de résumé non disponible.")

        st.subheader("❓ Question-Réponse sur les avis (RAG extractif)")
        question = st.text_input("Posez une question :",
                                  placeholder="Ex: Quels sont les principaux reproches ?",
                                  key="rag_question")
        if st.button("Obtenir une réponse", key="btn_qa"):
            if qa_pipe and question:
                texts = target_df['avis'].dropna().head(50).tolist()
                context = " ".join([str(t) for t in texts])[:2500]
                with st.spinner("Recherche de la réponse dans les avis..."):
                    try:
                        ans = qa_pipe(question=question, context=context)
                        st.success(f"**Réponse :** {ans['answer']}")
                        st.caption(f"Confiance : {ans['score']*100:.1f}%")
                    except Exception as e:
                        st.error(f"Erreur QA : {e}")
            else:
                st.error("Pipeline QA non disponible ou question vide.")
    else:
        st.warning("Transformers non installé ou dataset manquant.")
