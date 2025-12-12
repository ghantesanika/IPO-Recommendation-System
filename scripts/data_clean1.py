# scripts/data_clean1.py  (robust version)
import pandas as pd
import numpy as np
from pathlib import Path
import re

proj = Path(__file__).resolve().parent.parent
raw_path = proj / "data" / "raw" / "IPO_correct.csv"
out_path = proj / "data" / "processed" / "cleaned_dataset1.csv"

print("Loading:", raw_path)
df = pd.read_csv(raw_path, low_memory=False)

# helper: normalize column names to simple keys
def norm(col):
    if pd.isna(col):
        return ""
    s = str(col).strip().lower()
    s = re.sub(r'[\s\-\(\)_]+', ' ', s)
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    s = s.strip()
    return s

col_map = {norm(c): c for c in df.columns}

# helper to find best match from a list of possible names
def find_col(possible):
    for p in possible:
        key = norm(p)
        if key in col_map:
            return col_map[key]
    # try more fuzzy check: substring match
    for key, orig in col_map.items():
        for p in possible:
            if p.lower() in key:
                return orig
    return None

# list of canonical fields we want (with likely variants)
candidates = {
    "issue_open_date": ["issue open date","open date","open date of issue","issue open"],
    "issue_close_date": ["issue close date","close date","close date of issue","issue close"],
    "listing_date": ["listing date","listing"],
    "issue_price": ["issue price (rs)","issue price","issue price rs","issue price (rs)"],
    "upper_price": ["upper price band","upper price","upper band","upper band price"],
    "lower_price": ["lower price band","lower price","lower band","lower band price"],
    "issue_size_cr": ["total issue size (rs cr)","issue size (cr)","issue size (rs cr)","issue size"],
    "no_of_shares_cr": ["no. of shares (cr)","no of shares (cr)","no of shares","shares (cr)"],
    "sub_retail": ["subscription retail","retail subscription","retail sub","subscription (retail)"],
    "sub_qib": ["subscription qib","qib subscription","qib sub"],
    "sub_nii": ["subscription nii","nii subscription","non institutional investors","subscription non institutional"],
    "listing_gain": ["listing gain (%)","listing gain","listing gain %","listing gain percent"],
    "gmp": ["gmp","grey market premium","grey market premium (gmp)"]
}

# build a map from canonical name -> actual df column (or None)
mapped = {}
for k, candidates_list in candidates.items():
    mapped[k] = find_col(candidates_list)

print("\nColumn mapping found:")
for k,v in mapped.items():
    print(f"  {k:15s} -> {v!r}")

# For any missing columns, create defaults so script won't fail
for k in mapped:
    if mapped[k] is None:
        # create safe default column in df
        if k in ("issue_open_date","issue_close_date","listing_date"):
            df[k] = pd.NaT
            mapped[k] = k
        else:
            df[k] = 0
            mapped[k] = k

# Now use mapped columns to create standardized columns
# Convert dates
for key in ("issue_open_date","issue_close_date","listing_date"):
    col = mapped[key]
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# create Year from open date
df["Year"] = df[mapped["issue_open_date"]].dt.year

# Convert numeric columns
numeric_keys = ["issue_price","upper_price","lower_price","issue_size_cr","no_of_shares_cr",
                "sub_retail","sub_qib","sub_nii","listing_gain","gmp"]
for key in numeric_keys:
    col = mapped.get(key)
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df[col] = 0

# engineered features
df["IPO_Duration"] = (df[mapped["issue_close_date"]] - df[mapped["issue_open_date"]]).dt.days.fillna(0)
df["Price_Band_Width"] = (df[mapped["upper_price"]] - df[mapped["lower_price"]]).fillna(0)
# log issue size (avoid negative/zero)
df["log_Issue_Size"] = np.log1p(df[mapped["issue_size_cr"]].clip(lower=0))
# subscription combined strength
df["Sub_Strength"] = (df[mapped["sub_qib"]]*0.5 + df[mapped["sub_nii"]]*0.3 + df[mapped["sub_retail"]]*0.2)

# fill remaining NAs
df = df.fillna(0)

# Save cleaned df (use absolute path)
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8")
print("\nSaved cleaned file to:", out_path)
print("Rows:", len(df))
