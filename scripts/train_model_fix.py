# scripts/train_model_fix.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib, numpy as np, sys
from collections import Counter

proj = Path(__file__).resolve().parent.parent
data_path = proj / "data" / "processed" / "fe_dataset_labeled1.csv"
model_out = proj / "models" / "xgb_reco_fix.joblib"
encoder_out = proj / "models" / "label_encoder_fix.joblib"

print("Loading:", data_path)
df = pd.read_csv(data_path, low_memory=False)
print("Total rows:", len(df))

# Ensure Recommendation exists
if "Recommendation" not in df.columns:
    print("ERROR: Recommendation column missing. Run labeling script first.")
    sys.exit(1)

# Define candidate features (exclude any post-listing leakage features)
candidates = [
    "Issue Price (Rs)", "IPO_Duration",
    "log_Issue_Size", "Sub_Strength", "GMP"  # DO NOT include "Listing Gain (%)"
]

# If you have other pre-listing features, add them here.
features = [c for c in candidates if c in df.columns]
print("Raw available features:", features)

# Remove features that are constant or mostly zero
for f in features:
    nz = np.count_nonzero(df[f].fillna(0).values)
    pct_nonzero = nz / len(df)
    if pct_nonzero < 0.05:
        print(f"NOTE: feature '{f}' is mostly zero ({pct_nonzero:.2%} non-zero). Consider removing.")
# final features used
X = df[features].fillna(0)
y_raw = df["Recommendation"].astype(str).fillna("Neutral")

le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label classes:", le.classes_)
print("Label counts:", Counter(y_raw))

# 5-fold stratified CV (estimator pipeline)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric="mlogloss", verbosity=0))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Running 5-fold CV (stratified) ...")
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
print("CV accuracy: mean=%.4f std=%.4f" % (cv_scores.mean(), cv_scores.std()))

# Single train/test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training on", X_train.shape[0], "samples, testing on", X_test.shape[0], "samples")

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print("Classification report (on hold-out test):")
print(classification_report(y_test, pred, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

# Save model + encoder
model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_out)
joblib.dump(le, encoder_out)
print("Saved fixed model to:", model_out)
print("Saved label encoder to:", encoder_out)
