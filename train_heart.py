# train_heart.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib
import shap

DATA_PATH = "data/heart.csv"
os.makedirs("models", exist_ok=True)

TARGET_CANDIDATES = [
    "target", "num", "condition", "heartdisease", "heart_disease", "diagnosis", "label", "y"
]

def find_target_column(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in TARGET_CANDIDATES:
        if cand in cols_lower:
            actual = cols_lower[cand]
            print(f"Found target column by name: '{actual}'")
            return actual
    possible = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 5:
            possible.append((c, nunique))
    if len(possible) == 1:
        print(f"Using '{possible[0][0]}' as target (unique values = {possible[0][1]}).")
        return possible[0][0]
    elif len(possible) > 1:
        print("Multiple candidate columns with small unique counts found:", possible)
        for c,_ in possible:
            if any(k in c.lower() for k in ["target","cond","num","diag","disease","label"]):
                print("Selecting:", c)
                return c
        print("No clear candidate by name — using last column as fallback:", df.columns[-1])
        return df.columns[-1]
    print("No obvious target column found — using last column:", df.columns[-1])
    return df.columns[-1]


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Put your CSV at data/heart.csv")

    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset with shape:", df.shape)
    print(df.head())

    original_target_col = find_target_column(df)
    if original_target_col is None:
        raise ValueError("Could not determine target column. Please ensure csv has a target column.")

    y = df[original_target_col]
    # Convert to binary if numeric multiclass (e.g., 0..4)
    if y.dropna().dtype.kind in 'iuf':
        unique_vals = sorted(y.dropna().unique())
        if max(unique_vals) > 1 and min(unique_vals) >= 0:
            print(f"Target '{original_target_col}' appears multiclass (unique: {unique_vals}). Converting to binary (>0 => 1).")
            df['target'] = (df[original_target_col] > 0).astype(int)
        else:
            df['target'] = y.astype(int)
    else:
        df['target'] = pd.Categorical(y).codes

    # Drop the original target column to avoid leakage (IMPORTANT)
    if original_target_col != 'target' and original_target_col in df.columns:
        df = df.drop(columns=[original_target_col])
        print(f"Dropped original target column '{original_target_col}' from features to avoid leakage.")

    X = df.drop(columns=['target'])
    y = df['target'].astype(int)

    print("\nFinal target column used: 'target'")
    print("Feature columns:", list(X.columns))
    print("Target class distribution:\n", y.value_counts())

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            eval_metric="logloss",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    print("\nCross-val ROC-AUC:", cv_scores)
    print("Mean ROC-AUC:", cv_scores.mean())

    pipeline.fit(X_train, y_train)

    preds_proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    print("\nTest ROC-AUC:", roc_auc_score(y_test, preds_proba))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model
    joblib.dump(pipeline, "models/xgb_heart_pipeline.joblib")
    print("\nModel saved to models/xgb_heart_pipeline.joblib")

    # Create SHAP background sample
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    X_train_trans = scaler.transform(imputer.transform(X_train))
    bg = X_train_trans[np.random.choice(
        X_train_trans.shape[0], size=min(100, X_train_trans.shape[0]), replace=False
    )]

    joblib.dump(bg, "models/shap_bg_heart.joblib")
    print("SHAP background saved to models/shap_bg_heart.joblib")


if __name__ == "__main__":
    main()
