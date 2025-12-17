import pandas as pd
import numpy as np

# Load feature engineered dataset
INPUT_PATH = "data/processed/fe_dataset_labeled1.csv"
OUTPUT_PATH = "data/processed/fe_dataset_with_allotment.csv"

print("Loading:", INPUT_PATH)
df = pd.read_csv(INPUT_PATH)

# ---- IMPORTANT: Use Retail Subscription ----
# Change column name if yours is slightly different
retail_col = "Retail Subscription"

# Safety check
df[retail_col] = df[retail_col].replace(0, 0.1)

# Allotment Probability calculation
df["Allotment_Probability"] = np.where(
    df[retail_col] <= 1,
    1.0,
    1 / df[retail_col]
)

# Clamp between 0 and 1
df["Allotment_Probability"] = df["Allotment_Probability"].clip(0, 1)

print("Allotment Probability added successfully")

# Save new dataset
df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)
