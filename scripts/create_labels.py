# scripts/create_labels.py
import pandas as pd
from pathlib import Path

proj = Path(__file__).resolve().parent.parent
fe_path = proj / "data" / "processed" / "fe_dataset.csv"
out_path = proj / "data" / "processed" / "fe_dataset_labeled.csv"

df = pd.read_csv(fe_path, low_memory=False)

# simple heuristic labels (edit thresholds later)
def label_row(r):
    if r.get("Sub_Strength", 0) > 20 or r.get("GMP", 0) > 20:
        return "Apply"
    if r.get("Sub_Strength", 0) < 2:
        return "Avoid"
    return "Neutral"

df["Recommendation"] = df.apply(label_row, axis=1)
df.to_csv(out_path, index=False)
print("Saved labeled dataset:", out_path)
