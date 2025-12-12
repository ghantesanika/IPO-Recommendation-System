# scripts/train_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import numpy as np
import sys

proj = Path(__file__).resolve().parent.parent
data_path = proj / "data" / "processed" / "fe_dataset_labeled1.csv"   # adapt if your file name differs
model_out = proj / "models" / "xgb_reco_v1.joblib"
encoder_out = proj / "models" / "label_encoder.joblib"

print("Loading:", data_path)
df = pd.read_csv(data_path, low_memory=False)

print("Total rows:", len(df))
print("Columns available:", list(df.columns))

# Candidate features you want to use (edit if needed)
candidates = [
    "Issue Price (Rs)", "IPO_Duration", "Price_Band_Width",
    "log_Issue_Size", "Sub_Strength", "GMP", "Listing Gain (%)"
]

# Make sure every candidate exists in df; if not, create safe fallback columns
for c in candidates:
    if c not in df.columns:
        print(f"WARNING: feature '{c}' not found in dataset -> creating fallback column filled with 0")
        df[c] = 0

# Now select features that exist (they all will because we created fallbacks)
features = [c for c in candidates if c in df.columns]
print("Final features used:", features)

# Build X,y
X = df[features].fillna(0)
if "Recommendation" not in df.columns:
    print("ERROR: 'Recommendation' column not found in dataset. Run labeling script first.")
    sys.exit(1)

y_raw = df["Recommendation"].astype(str).fillna("Neutral")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label classes:", le.classes_)
from collections import Counter
print("Label counts:", Counter(y_raw))

# Train/test split with stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline and classifier
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric="mlogloss"))
])

# Optional: quick 5-fold CV to get a better estimate
try:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    print("5-fold CV accuracy: mean=%.4f std=%.4f" % (cv_scores.mean(), cv_scores.std()))
except Exception as e:
    print("CV failed:", e)

# Fit the model
print("Training model on", X_train.shape[0], "samples...")
pipe.fit(X_train, y_train)

# Evaluate
pred = pipe.predict(X_test)
print("Classification report:")
print(classification_report(y_test, pred, target_names=le.classes_))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred))

# Save
model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_out)
joblib.dump(le, encoder_out)
print("Saved model to:", model_out)
print("Saved label encoder to:", encoder_out)
