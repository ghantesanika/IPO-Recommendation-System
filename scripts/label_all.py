import pandas as pd
from pathlib import Path
import numpy as np

proj = Path(__file__).resolve().parent.parent
fe_path = proj / "data" / "processed" / "fe_dataset1.csv"
out_path = proj / "data" / "processed" / "fe_dataset_labeled1.csv"

print("Loading:", fe_path)
df = pd.read_csv(fe_path)

def auto_label(row):
    lg = row["Listing Gain (%)"]
    sub = row["Sub_Strength"]
    gmp = row.get("GMP", 0)

    # Priority 1: Listing Gain
    if lg >= 10:
        return "Apply"
    elif lg <= 0:
        return "Avoid"

    # Priority 2: Subscription strength
    if sub >= 20:
        return "Apply"
    elif sub <= 2:
        return "Avoid"

    # Priority 3: GMP fallback
    if gmp >= 20:
        return "Apply"
    elif gmp <= -5:
        return "Avoid"

    return "Neutral"

df["Recommendation"] = df.apply(auto_label, axis=1)

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print("Labels created successfully!")
print(df["Recommendation"].value_counts())
