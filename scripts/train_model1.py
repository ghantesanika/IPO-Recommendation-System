# scripts/train_model.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

proj = Path(__file__).resolve().parent.parent
data_path = proj / "data" / "processed" / "fe_dataset_labeled1.csv"
model_out = proj / "models" / "xgb_reco_v1.joblib"
encoder_out = proj / "models" / "label_encoder.joblib"

print("Loading:", data_path)
df = pd.read_csv(data_path, low_memory=False)

# Choose features (adjust to your dataset)
candidates = ["Issue Price (Rs)", "IPO_Duration", "Price_Band_Width",
              "log_Issue_Size", "Sub_Strength", "GMP_day_before", "Listing Gain (%)"]
features = [c for c in candidates if c in df.columns]
if not features:
    raise SystemExit("No features found. Check feature names in fe_dataset_labeled1.csv")

print("Using features:", features)

X = df[features].fillna(0)
y_raw = df["Recommendation"].astype(str).fillna("Neutral")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label classes:", le.classes_)
print("Counts:", dict(zip(le.classes_, np.bincount(y))))

# Train/test split with stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Basic XGBoost with early stopping via CV (if desired)
# If classes are imbalanced, XGBoost can use scale_pos_weight, but for multi-class we can use class_weight alternatives.
clf = XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric="mlogloss", verbosity=1)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", clf)
])

# Cross-validation (stratified) to get a more reliable performance estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print("5-fold CV accuracy: mean=%.4f std=%.4f" % (cv_scores.mean(), cv_scores.std()))

# Fit on training data
pipe.fit(X_train, y_train)

# Evaluate on test set
pred = pipe.predict(X_test)
print("Classification report:")
print(classification_report(y_test, pred, target_names=le.classes_))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred))

# Save model + encoder
model_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, model_out)
joblib.dump(le, encoder_out)
print("Saved model to:", model_out)
print("Saved label encoder to:", encoder_out)
