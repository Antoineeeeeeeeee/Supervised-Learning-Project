"""
summary_translation.py
Generates:
  - data/processed/insurer_summaries.csv  : one NLP-generated summary per insurer
  - updated cleaned_dataset.csv with a new column `avis_en` (English translation via argostranslate or deep_translator fallback)
"""
import os
import pandas as pd

print("Loading cleaned dataset...")
df = pd.read_csv(os.path.join("data", "processed", "cleaned_dataset.csv"))
out_dir = os.path.join("data", "processed")

# ─────────────────────────────────────────────────────────────
# 1. SUMMARIZATION – one summary per insurer (via Transformers)
# ─────────────────────────────────────────────────────────────
print("\n[1/2] Generating per-insurer summaries with a transformer summarizer...")
try:
    from transformers import pipeline
    # T5-based models require `text2text-generation` pipeline task (not "summarization")
    sum_pipe = pipeline("text2text-generation", model="plguillou/t5-base-fr-sum-cnndm")

    summaries = []
    for assureur, grp in df.groupby("assureur"):
        sample_texts = grp["avis_clean"].dropna().head(15).tolist()
        combined = " ".join([str(t) for t in sample_texts])[:1200]
        try:
            result = sum_pipe(combined, max_length=120, min_length=20, do_sample=False)
            # text2text-generation returns `generated_text`, not `summary_text`
            summary_text = result[0]["generated_text"]
        except Exception as e:
            summary_text = f"[Erreur: {e}]"
        summaries.append({"assureur": assureur, "summary": summary_text})

    df_summaries = pd.DataFrame(summaries)
    df_summaries.to_csv(os.path.join(out_dir, "insurer_summaries.csv"), index=False, encoding="utf-8")
    print(f"  -> Saved summaries to data/processed/insurer_summaries.csv ({len(df_summaries)} rows)")

except ImportError:
    print("  Transformers not installed. Skipping summarization.")

# ─────────────────────────────────────────────────────────────
# 2. TRANSLATION – add `avis_en` column (FR -> EN) on a sample
# ─────────────────────────────────────────────────────────────
print("\n[2/2] Translating a sample of reviews (FR -> EN)...")

TRANSLATION_SAMPLE = 500   # translate the first N rows to avoid long runtimes

translated_col = [""] * len(df)
rows_to_translate = min(TRANSLATION_SAMPLE, len(df))

try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="fr", target="en")

    for i in range(rows_to_translate):
        raw_text = str(df.iloc[i]["avis_clean"])[:400]
        try:
            translated_col[i] = translator.translate(raw_text)
        except Exception:
            translated_col[i] = ""
    print(f"  -> Translated {rows_to_translate} rows via deep_translator (GoogleTranslator).")

except ImportError:
    print("  deep_translator not installed. Trying argostranslate as fallback...")
    try:
        from argostranslate import package, translate
        package.update_package_index()
        available_packages = package.get_available_packages()
        pkg = next(
            (p for p in available_packages if p.from_code == "fr" and p.to_code == "en"),
            None
        )
        if pkg:
            package.install_from_path(pkg.download())
        installed_languages = translate.get_installed_languages()
        from_lang = next(l for l in installed_languages if l.code == "fr")
        to_lang = next(l for l in installed_languages if l.code == "en")
        translation_fn = from_lang.get_translation(to_lang)

        for i in range(rows_to_translate):
            raw_text = str(df.iloc[i]["avis_clean"])[:300]
            try:
                translated_col[i] = translation_fn.translate(raw_text)
            except Exception:
                translated_col[i] = ""
        print(f"  -> Translated {rows_to_translate} rows via argostranslate.")
    except Exception as e:
        print(f"  Argostranslate also failed: {e}. Column `avis_en` will be empty.")

df["avis_en"] = translated_col

df.to_csv(os.path.join(out_dir, "cleaned_dataset.csv"), index=False)
print(f"\nUpdated cleaned_dataset.csv with `avis_en` column. Done.")
