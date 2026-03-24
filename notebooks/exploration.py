
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_cleaning import load_multiple_excels, preview_data, frequent_words

def main():
    folder = r"data/raw/Traduction avis clients"
    df = load_multiple_excels(folder)
    print("Aperçu des données:")
    print(preview_data(df, 10))
    print("\nMots les plus fréquents:")
    print(frequent_words(df, "avis_en", 20))

if __name__ == "__main__":
    main()
