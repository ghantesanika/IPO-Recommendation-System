from pathlib import Path
import pandas as pd
import numpy as np

# Load raw dataset
#df = pd.read_csv("../data/Raw/IPO_correct.csv", low_memory=False)
BASE = Path(__file__).resolve().parent.parent  # IPO_Pro_final/scripts -> parent is project root
data_path= BASE / "data" / "raw" / "IPO_correct.csv"
print("Trying to load:", data_path)
df = pd.read_csv(data_path, low_memory=False)
#df=pd.read_csv("../Data/Raw/IPO_correct.csv")




# -------------------------------
# 1. Standardize IPO Name
# -------------------------------
df["IPO Name"] = df["IPO Name"].astype(str).str.strip().str.title()

# -------------------------------
# 2. Convert date columns
# -------------------------------
date_cols = ["Issue Open Date", "Issue Close Date", "Listing Date"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -------------------------------
# 3. Create 'Year' column from Issue Open Date
# -------------------------------
df["Year"] = df["Issue Open Date"].dt.year

# -------------------------------
# 4. Convert numeric columns
# -------------------------------
numeric_cols = [
    "Issue Price (Rs)",
    "Listing Price (Rs)",
    "Listing Gain (%)",
    "Upper Price",
    "Lower Price",
    "Issue Size (Cr)"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 5. Add Feature: IPO Duration (Open → Close)
# -------------------------------
if "Issue Open Date" in df.columns and "Issue Close Date" in df.columns:
    df["IPO_Duration"] = (df["Issue Close Date"] - df["Issue Open Date"]).dt.days

# -------------------------------
# 6. Add Feature: Price Band Width
# -------------------------------
if "Upper Price" in df.columns and "Lower Price" in df.columns:
    df["Price_Band_Width"] = df["Upper Price"] - df["Lower Price"]

# -------------------------------
# 7. Handle Missing Values
# -------------------------------
df.fillna(0, inplace=True)

# -------------------------------
# 8. Save cleaned dataset
# -------------------------------
#output_path = "../data/processed/cleaned_dataset.csv"
#df.to_csv(output_path, index=False)

#print("Cleaning completed successfully!")
#print("Saved cleaned file to:", output_path)

# get project root reliably (script is in scripts/ so parent is project root)
script_dir = Path(__file__).resolve().parent      # .../IPO_Pro_final/scripts
project_root = script_dir.parent                   # .../IPO_Pro_final

# prepare output directory and file
out_dir = project_root / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)        # creates folder if missing

output_path = out_dir / "cleaned_dataset1.csv"

# save DataFrame (df) to this path
df.to_csv(output_path, index=False, encoding="utf-8")

print("Cleaning completed successfully!")
print("Saved cleaned file to:", output_path)
print("Absolute path exists:", output_path.exists())