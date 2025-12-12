# scripts/train_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

proj = Path(__file__).resolve().parent.parent
data_path = proj / "data" / "processed" / "fe_dataset_labeled.csv"
model_out = proj / "models" / "xgb_reco_v1.joblib"
encoder_out = proj / "models" / "label_encoder.joblib"

print("Loading:", data_path)
df = pd.read_csv(data_path, low_memory=False)

# Choose features (adjust according to your fe_dataset)
candidates = ["Issue Price (Rs)", "IPO_Duration", "Price_Band_Width",
              "log_Issue_Size", "Sub_Strength","GMP"]
features = [c for c in candidates if c in df.columns]
if not features:
    raise SystemExit("No features found. Check feature names in fe_dataset_labeled.csv")

print("Using features:", features)

X = df[features].fillna(0)
y_raw = df["Recommendation"].astype(str).fillna("Neutral")

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(y_raw)  # e.g. ['Apply','Avoid','Neutral'] -> [0,1,2] (mapping printed below)

print("Label classes:", le.classes_)
print("Mapping (class -> int):")
for cls, val in zip(le.classes_, range(len(le.classes_))):
    print(f"  {cls} -> {val}")

# Train/test split (stratify on encoded labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline with scaler + XGBoost
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric="mlogloss", verbosity=1))
])

print("Training...")
pipe.fit(X_train, y_train)

# Eval
pred = pipe.predict(X_test)
print("Classification report (numeric labels):")
print(classification_report(y_test, pred, target_names=le.classes_))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred))

# Save model + encoder
model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_out)
joblib.dump(le, encoder_out)
print("Saved model to:", model_out)
print("Saved label encoder to:", encoder_out)
