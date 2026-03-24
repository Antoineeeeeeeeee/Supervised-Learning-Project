import os
import pandas as pd
import glob

data_dir = r"c:\Users\antoi\OneDrive\Bureau\ESILV !!\A4\S8\NLP\Supervised_Learning_Project_NLP2\Traduction avis clients\Traduction avis clients"
files = glob.glob(os.path.join(data_dir, "*.xlsx"))
print(f"Found {len(files)} files.")

if files:
    try:
        df = pd.read_excel(files[0])
        print("File:", os.path.basename(files[0]))
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("\nFirst 2 rows:")
        print(df.head(2).to_string())
    except Exception as e:
        print("Error reading first file:", e)
        
    if len(files) > 1:
        try:
            df2 = pd.read_excel(files[1])
            print("\nFile:", os.path.basename(files[1]))
            print("Columns match?", df.columns.tolist() == df2.columns.tolist())
        except Exception as e:
            print("Error reading second file:", e)
