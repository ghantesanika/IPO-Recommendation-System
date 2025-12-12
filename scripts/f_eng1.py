# scripts/feature_engineering_robust.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

proj = Path(__file__).resolve().parent.parent
in_path = proj / "data" / "processed" / "cleaned_dataset1.csv"   # change filename if different
out_path = proj / "data" / "processed" / "fe_dataset1.csv"

print("Loading:", in_path)
df = pd.read_csv(in_path, low_memory=False)

# helper to normalize names
def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\s\-\(\)_]+', ' ', s)
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    return s

cols = {norm(c): c for c in df.columns}

def find_column(possible_names):
    # return actual column name if any variant matches
    for p in possible_names:
        key = norm(p)
        if key in cols:
            return cols[key]
    # try substring matching
    for key_norm, real in cols.items():
        for p in possible_names:
            if p.lower() in key_norm:
                return real
    return None

# candidate lists (common variants)
upper_candidates = ["upper price band", "upper price", "upper band", "upper band price", "upperprice", "upper_price"]
lower_candidates = ["lower price band", "lower price", "lower band", "lower band price", "lowerprice", "lower_price"]
open_date_candidates = ["issue open date","open date","open date of issue","issue open"]
close_date_candidates = ["issue close date","close date","close date of issue","issue close"]
issue_size_candidates = ["issue size (cr)","total issue size (rs cr)","issue size", "issue size rs cr"]
issue_price_candidates = ["issue price (rs)","issue price","issue price rs"]

# find actual column names
upper_col = find_column(upper_candidates)
lower_col = find_column(lower_candidates)
open_col = find_column(open_date_candidates)
close_col = find_column(close_date_candidates)
issue_size_col = find_column(issue_size_candidates)
issue_price_col = find_column(issue_price_candidates)

print("Found mapping:")
print("  upper_col ->", upper_col)
print("  lower_col ->", lower_col)
print("  open_col  ->", open_col)
print("  close_col ->", close_col)
print("  issue_size_col ->", issue_size_col)
print("  issue_price_col ->", issue_price_col)

# Create safe numeric columns if missing
if upper_col is None:
    df["__upper_price"] = 0
    upper_col = "__upper_price"
if lower_col is None:
    df["__lower_price"] = 0
    lower_col = "__lower_price"
if open_col is None:
    df["__open_date"] = pd.NaT
    open_col = "__open_date"
else:
    df[open_col] = pd.to_datetime(df[open_col], errors="coerce")
if close_col is None:
    df["__close_date"] = pd.NaT
    close_col = "__close_date"
else:
    df[close_col] = pd.to_datetime(df[close_col], errors="coerce")
if issue_size_col is None:
    df["__issue_size"] = 0
    issue_size_col = "__issue_size"
else:
    df[issue_size_col] = pd.to_numeric(df[issue_size_col], errors="coerce").fillna(0)
if issue_price_col is None:
    df["__issue_price"] = 0
    issue_price_col = "__issue_price"
else:
    df[issue_price_col] = pd.to_numeric(df[issue_price_col], errors="coerce").fillna(0)

# Engineer features
df["Price_Band_Width"] = (pd.to_numeric(df[upper_col], errors="coerce").fillna(0)
                          - pd.to_numeric(df[lower_col], errors="coerce").fillna(0))

df["IPO_Duration"] = (pd.to_datetime(df[close_col], errors="coerce") - pd.to_datetime(df[open_col], errors="coerce")).dt.days.fillna(0)

df["log_Issue_Size"] = np.log1p(pd.to_numeric(df[issue_size_col], errors="coerce").fillna(0))

# Subscription strength: try to find retail/qib/nii columns automatically
sub_ret = find_column(["subscription retail","retail subscription","retail sub","sub retail","subscription (retail)"])
sub_qib = find_column(["subscription qib","qib subscription","sub qib","subscription (qib)"])
sub_nii = find_column(["subscription nii","nii subscription","sub nii","non institutional subscription","subscription (nii)"])

print("Subscription columns found:", sub_ret, sub_qib, sub_nii)

def to_num_col(col):
    if col is None:
        return pd.Series([0]*len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0)

ret_series = to_num_col(sub_ret)
qib_series = to_num_col(sub_qib)
nii_series = to_num_col(sub_nii)
df["Sub_Strength"] = 0.5*qib_series + 0.3*nii_series + 0.2*ret_series

# Ensure no-nulls and types
df["Price_Band_Width"] = df["Price_Band_Width"].fillna(0)
df["IPO_Duration"] = df["IPO_Duration"].fillna(0)
df["log_Issue_Size"] = df["log_Issue_Size"].fillna(0)
df["Sub_Strength"] = df["Sub_Strength"].fillna(0)

# Save
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8")
print("Saved FE dataset to:", out_path)
print("Rows:", len(df))
