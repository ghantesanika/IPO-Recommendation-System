# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path

proj = Path(__file__).resolve().parent.parent
clean_path = proj / "data" / "processed" / "cleaned_dataset.csv"
out_path = proj / "data" / "processed" / "fe_dataset.csv"

print("Loading:", clean_path)
df = pd.read_csv(clean_path, low_memory=False)

# --- Example feature creation (add/adjust columns to match your CSV names) ---
# 1) Year (if missing)
if "Year" not in df.columns and "Issue Open Date" in df.columns:
    df["Issue Open Date"] = pd.to_datetime(df["Issue Open Date"], errors="coerce")
    df["Year"] = df["Issue Open Date"].dt.year

# 2) Price band width
if "Upper Price" in df.columns and "Lower Price" in df.columns:
    df["Price_Band_Width"] = df["Upper Price"] - df["Lower Price"]

# 3) IPO duration (days)
if "Issue Open Date" in df.columns and "Issue Close Date" in df.columns:
    df["Issue Open Date"] = pd.to_datetime(df["Issue Open Date"], errors="coerce")
    df["Issue Close Date"] = pd.to_datetime(df["Issue Close Date"], errors="coerce")
    df["IPO_Duration"] = (df["Issue Close Date"] - df["Issue Open Date"]).dt.days.fillna(0)

# 4) Log transforms for skewed numeric columns (if present)
for col in ["Issue Size (Cr)", "Issue Price (Rs)", "Listing Price (Rs)"]:
    if col in df.columns:
        df[f"log_{col.replace(' ','_').replace('(','').replace(')','')}"] = np.log1p(df[col].fillna(0))

# 5) Subscription features cleanup (if present, e.g., sub_retail_2015 etc.)
# Convert any subscription-like columns to numeric (remove 'x' or 'times')
for c in df.columns:
    if isinstance(c, str) and ("sub" in c.lower() or "subscription" in c.lower()):
        df[c] = df[c].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# 6) GMP placeholder (0 for now) — will fill later after scraping
#df["GMP"] = df.get("GMP", 0).fillna(0)


if "GMP" not in df.columns:
    df["GMP"] = 0
else:
    df["GMP"] = df["GMP"].fillna(0)


# 7) Combined score example
# weight QIB (0.5), NII (0.3), Retail (0.2) if these columns exist
ret = next((col for col in df.columns if "retail" in col.lower()), None)
nii = next((col for col in df.columns if "nii" in col.lower()), None)
qib = next((col for col in df.columns if "qib" in col.lower()), None)

if ret or nii or qib:
    df["Sub_Strength"] = 0
    if qib: df["Sub_Strength"] += 0.5 * df[qib]
    if nii: df["Sub_Strength"] += 0.3 * df[nii]
    if ret: df["Sub_Strength"] += 0.2 * df[ret]
else:
    df["Sub_Strength"] = 0

# 8) Fill NA and basic cleanup
df.fillna(0, inplace=True)

# Save
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print("Saved FE dataset:", out_path)
