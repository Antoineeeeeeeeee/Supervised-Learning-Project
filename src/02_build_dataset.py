import os
import pandas as pd
import glob
import io

data_dir = r"c:\Users\antoi\OneDrive\Bureau\ESILV !!\A4\S8\NLP\Supervised_Learning_Project_NLP2\Traduction avis clients\Traduction avis clients"
files = glob.glob(os.path.join(data_dir, "*.xlsx"))
print(f"Loading {len(files)} files...")

dfs = []
for file in files:
    try:
        df = pd.read_excel(file)
        # some files may have a slightly different structure or extra rows, let's keep it simple
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if dfs:
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Let's capture info
    buf = io.StringIO()
    full_df.info(buf=buf)
    info_str = buf.getvalue()
    
    # Capture first few rows
    head_str = full_df.head(5).to_string()
    
    with open("dataset_info.txt", "w", encoding="utf-8") as f:
        f.write(info_str + "\n\n" + head_str)
        
    out_path = os.path.join("data", "raw", "full_dataset.csv")
    full_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved dataset to {out_path}")
