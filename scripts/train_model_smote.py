# scripts/train_model_smote_safe.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
from collections import Counter

# imblearn imports
from imblearn.over_sampling import SMOTE, RandomOverSampler

proj = Path(__file__).resolve().parent.parent
data_path = proj / "data" / "processed" / "fe_dataset_labeled1.csv"   # update if different
print("Loading:", data_path)
df = pd.read_csv(data_path, low_memory=False)

# Try to detect target column
target_candidates = ["Recommendation", "RecommendationLabel", "Label", "Recommendation "]
target_col = None
for c in target_candidates:
    if c in df.columns:
        target_col = c
        break
# fallback: if only one column with small unique values, list all columns and ask user
if target_col is None:
    # try to find a column with 2+ unique string labels
    for c in df.columns:
        if df[c].dtype == object or df[c].nunique() <= 10:
            if df[c].nunique() >= 2:
                target_col = c
                break

if target_col is None:
    raise SystemExit("Could not find target column. Please ensure you have a 'Recommendation' or 'Label' column.")

print("Using target column:", target_col)
print("Unique values and counts:")
print(df[target_col].value_counts())

# select features - use only pre-listing, non-leaking features from your dataset
#candidates = [
 #   "upper_price", "lower_price", "no_of_shares_cr", "IPO_Duration",
  #  "Price_Band_Width", "log_Issue_Size", "Year"
#]
candidates = [
    "Issue Price (Rs)", "IPO_Duration",
    "log_Issue_Size", "Sub_Strength", "GMP"  # DO NOT include "Listing Gain (%)"
]
# add any other safe features present
features = [c for c in candidates if c in df.columns]
print("Features found and will be used:", features)
if not features:
    raise SystemExit("No features found. Please check feature names in fe_dataset_labeled1.csv")

X = df[features].fillna(0)
y_raw = df[target_col].astype(str).fillna("Neutral")

# encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Encoded label classes:", le.classes_)
print("Label counts (raw):", dict(Counter(y_raw)))

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train label distribution:", dict(Counter(y_train)))
print("Test label distribution:", dict(Counter(y_test)))

# Decide resampler:
unique, counts = np.unique(y_train, return_counts=True)
min_count = counts.min()
n_classes = len(unique)
print("Number of classes in train:", n_classes, "min class count:", min_count)

resampler = None
use_smote = False
if n_classes < 2:
    raise SystemExit("Training set has fewer than 2 classes. Cannot train classifier.")
# choose k_neighbors for SMOTE safely
k_neighbors = 3
if min_count <= 1:
    # SMOTE impossible if minority class has only 1 sample
    print("Minority class too small for SMOTE (<=1). Falling back to RandomOverSampler.")
    resampler = RandomOverSampler(random_state=42)
else:
    # set k_neighbors to min(3, min_count-1)
    k = min(k_neighbors, min_count - 1)
    if k < 1:
        print("k would be <1; using RandomOverSampler")
        resampler = RandomOverSampler(random_state=42)
    else:
        try:
            resampler = SMOTE(random_state=42, k_neighbors=k)
            use_smote = True
            print("Using SMOTE with k_neighbors =", k)
        except Exception as e:
            print("SMOTE init failed:", e)
            resampler = RandomOverSampler(random_state=42)

# apply resampler only on training data
X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
print("After resampling train label counts:", dict(Counter(le.inverse_transform(y_train_res))))

# train model
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    max_depth=4,
    learning_rate=0.1,
    n_estimators=200,
    use_label_encoder=False,
    verbosity=0
)

print("Training model on", X_train_res.shape[0], "samples ...")
model.fit(X_train_res, y_train_res)

# eval on untouched test set
pred = model.predict(X_test)
print("\nClassification report (test):")
print(classification_report(y_test, pred, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

# save model & encoder
out_model = proj / "models" / "xgb_smote_safe.joblib"
out_encoder = proj / "models" / "label_encoder_smote_safe.joblib"
out_model.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out_model)
joblib.dump(le, out_encoder)
print("Saved model to:", out_model)
print("Saved encoder to:", out_encoder)
