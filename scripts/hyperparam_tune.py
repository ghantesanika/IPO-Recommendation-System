# scripts/hyperparam_tune_safe.py
"""
Robust Hyperparameter tuning script (SMOTE inside CV, safe saves).
- Set n_jobs=1 to avoid loky/resource-tracker issues on some Windows setups.
- Uses RandomizedSearchCV with SMOTE in pipeline (no leakage).
- Creates fallback columns if expected features missing.
"""

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import joblib
import json

# ------------- Config -------------
proj = Path(__file__).resolve().parent.parent
DATA_PATH = proj / "data" / "processed" / "fe_dataset_labeled1.csv"   # update if different
OUT_MODEL = proj / "models" / "xgb_tuned_smote_safe1.joblib"
OUT_ENCODER = proj / "models" / "label_encoder_tuned_safe1.joblib"
N_ITER = 40                 # number of random search iterations (increase if you have time)
CV_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = 1                  # <--- set to 1 to avoid loky issues; set >1 if your env is OK

# Candidate (safe) feature names (common variants). The script will pick those present.
SAFE_FEATURE_CANDIDATES = [
    "Issue Price (Rs)", "Issue Price", "Issue_Price", "Issue Price Rs",
    "IPO_Duration", "Price_Band_Width", "log_Issue_Size", "log_Issue_Size",
    "Sub_Strength", "log_issue_size", "Price_Band_Width"
]

# ------------- Load -------------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows:", len(df))
print("Columns found:", len(df.columns))

# ------------- Detect target column -------------
# Prefer 'Recommendation' else fallback to 'Label' or 'RecommendationLabel'
target_candidates = ["Recommendation", "RecommendationLabel", "Label", "Recommendation "]
target_col = None
for c in target_candidates:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    # try to find a likely categorical target with >=2 classes
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() >= 2:
            target_col = c
            break

if target_col is None:
    raise SystemExit("ERROR: Could not find a target column. Ensure fe_dataset_labeled1.csv has Recommendation/Label.")

print("Using target column:", target_col)
print(df[target_col].value_counts())

# ------------- Select features (safe only) -------------
# find whichever of SAFE_FEATURE_CANDIDATES appear in df
features = []
for cand in SAFE_FEATURE_CANDIDATES:
    if cand in df.columns and cand not in features:
        features.append(cand)

# if no SAFE features found, try to choose numeric features that are not the target and not obviously post-listing
if not features:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target_col]
    # prefer known names if present
    prefer = ["Issue Price (Rs)", "IPO_Duration", "log_Issue_Size", "Price_Band_Width", "Sub_Strength"]
    for p in prefer:
        if p in numeric_cols and p not in features:
            features.append(p)
    # if still none, pick top 6 numeric columns excluding listing gain or post-listing columns
    excluded_keywords = ["Listing", "Gain", "Listing Gain", "Allotment", "list"]
    numeric_candidates = [c for c in numeric_cols if not any(k.lower() in c.lower() for k in excluded_keywords)]
    features.extend(numeric_candidates[:6])

# Create fallback numeric columns for any missing standard features to avoid KeyErrors later
for f in features:
    if f not in df.columns:
        df[f] = 0

print("Final features used:", features)
if len(features) == 0:
    raise SystemExit("ERROR: No features selected. Inspect fe_dataset_labeled.csv and SAFE_FEATURE_CANDIDATES.")

# ------------- Prepare X, y -------------
X = df[features].fillna(0)
y_raw = df[target_col].astype(str).fillna("Neutral")

le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Encoded classes:", le.classes_, "Counts:", dict(zip(le.classes_, np.bincount(y))))

# ------------- Build pipeline (SMOTE -> Scaler -> XGB) -------------
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
scaler = StandardScaler()
xgb = XGBClassifier(objective="multi:softprob", use_label_encoder=False, eval_metric="mlogloss", verbosity=0, random_state=RANDOM_STATE)

pipe = ImbPipeline([
    ("smote", smote),
    ("scaler", scaler),
    ("xgb", xgb)
])

# ------------- Hyperparameter search space -------------
param_dist = {
    "xgb__n_estimators": randint(100, 700),
    "xgb__max_depth": randint(3, 8),
    "xgb__learning_rate": uniform(0.01, 0.3),
    "xgb__subsample": uniform(0.6, 0.4),        # 0.6 - 1.0
    "xgb__colsample_bytree": uniform(0.5, 0.5), # 0.5 - 1.0
    "xgb__min_child_weight": randint(1, 10),
    "xgb__gamma": uniform(0, 3)
}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,            # <--- important: keep 1 to avoid loky issues
    return_train_score=False
)

# ------------- Run search -------------
start = time.time()
print("Starting RandomizedSearchCV with n_iter =", N_ITER, "n_jobs =", N_JOBS)
rs.fit(X, y)
end = time.time()
print(f"RandomizedSearchCV done in {(end-start)/60:.2f} minutes")

# ------------- Results -------------
best_score = rs.best_score_
best_params = rs.best_params_
print("\nBEST CV f1_macro: %.4f" % best_score)
print("Best parameters:")
print(json.dumps(best_params, indent=2))

# ------------- Evaluate best on hold-out 20% split -------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
best_est = rs.best_estimator_

# fit best_est on training fold (it contains SMOTE inside pipeline)
best_est.fit(X_train, y_train)

y_pred = best_est.predict(X_test)
print("\nClassification report (hold-out test):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ------------- Safe save -------------
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
try:
    joblib.dump(best_est, OUT_MODEL)
    joblib.dump(le, OUT_ENCODER)
    print("Saved tuned model to:", OUT_MODEL)
    print("Saved label encoder to:", OUT_ENCODER)
except Exception as e:
    print("Warning: joblib.dump raised an exception during cleanup/save:", repr(e))
    # try an alternative safe save (use cloudpickle via joblib)
    try:
        import cloudpickle
        with open(OUT_MODEL, "wb") as f:
            cloudpickle.dump(best_est, f)
        with open(OUT_ENCODER, "wb") as f:
            cloudpickle.dump(le, f)
        print("Saved model & encoder via cloudpickle fallback.")
    except Exception as e2:
        print("Fallback save failed too:", repr(e2))
        print("You can still access the best estimator via 'rs.best_estimator_' in memory until script ends.")

print("\nDone.")
