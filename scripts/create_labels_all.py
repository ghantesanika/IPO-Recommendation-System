# scripts/create_labels_all.py
import pandas as pd
from pathlib import Path
import numpy as np

proj = Path(__file__).resolve().parent.parent
fe_path = proj / "data" / "processed" / "fe_dataset.csv"
out_path = proj / "data" / "processed" / "fe_dataset_labeled1.csv"

print("Loading:", fe_path)
df = pd.read_csv(fe_path, low_memory=False)

# --- Strategy: create automatic labels for all IPOs ---
# Primary signal: Listing Gain (%) when available.
# Secondary signal: Sub_Strength (subscription score)
# Tertiary fallback: GMP_day_before (if available).
#
# Rules (you can tweak the thresholds):
# - If Listing Gain >= +10%  => Apply
# - Else if Listing Gain <= 0% => Avoid
# - Else if Sub_Strength >= 20 => Apply
# - Else if Sub_Strength <= 2 => Avoid
# - Else => Neutral

def auto_label(row):
    lg = row.get("Listing Gain (%)")
    sub = row.get("Sub_Strength", 0)
    gmp = row.get("GMP", 0)

    # prefer numeric listing gain if present and not zero-filled
    if pd.notna(lg) and lg != 0:
        try:
            lg = float(lg)
            if lg >= 10:
                return "Apply"
            if lg <= 0:
                return "Avoid"
        except:
            pass

    # fallback to subscription strength
    try:
        sub = float(sub)
        if sub >= 20:
            return "Apply"
        if sub <= 2:
            return "Avoid"
    except:
        pass

    # last fallback: GMP
    try:
        gmp = float(gmp)
        if gmp >= 20:
            return "Apply"
        if gmp <= -5:
            return "Avoid"
    except:
        pass

    return "Neutral"

# Apply to every row
df["Recommendation"] = df.apply(auto_label, axis=1)

# Save and show summary counts
df.to_csv(out_path, index=False, encoding="utf-8")
counts = df["Recommendation"].value_counts()
print("Saved labeled dataset:", out_path)
print("Label distribution:\n", counts)
