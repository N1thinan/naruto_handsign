#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
#  step2_train_model.py
#
#  Trains an ensemble (RandomForest + XGBoost + CatBoost) with
#  soft voting. Uses landmarks_train.csv for training and
#  landmarks_test.csv for held-out evaluation.
#
#  Run:
#    python step2_train_model.py
# ─────────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report, confusion_matrix, accuracy_score
from xgboost  import XGBClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import HAND_SIGNS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/landmarks_train.csv")
    p.add_argument("--test_csv",  default="data/landmarks_test.csv")
    p.add_argument("--output",    default="models/ensemble.pkl")
    p.add_argument("--cv",        type=int, default=5)
    return p.parse_args()


def load_csv(path: str):
    df = pd.read_csv(path)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y  = df["label"].values
    return X, y


def build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_seed=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("cat", cat)],
        voting="soft",
        n_jobs=1,
    )


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    print("── Loading data ──────────────────────────────────────")
    X_train, y_train_raw = load_csv(args.train_csv)
    X_test,  y_test_raw  = load_csv(args.test_csv)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Test : {len(X_test)}  samples")
    print(f"  Features: {X_train.shape[1]}")

    # Fit encoder on known classes (all 13) so indices are stable
    le = LabelEncoder()
    le.fit(HAND_SIGNS)
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw)
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}\n")

    # ── Cross-validation on train set ─────────────────────────
    print(f"── {args.cv}-fold CV on train set ────────────────────────")
    ensemble = build_ensemble()
    cv        = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train, y_train,
                                cv=cv, scoring="accuracy", n_jobs=1)
    print(f"  Fold accuracies : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean ± Std      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # ── Train on full train set ────────────────────────────────
    print("── Training on full train set ────────────────────────")
    ensemble.fit(X_train, y_train)

    # ── Evaluate on held-out test set ─────────────────────────
    print("── Test set evaluation ───────────────────────────────")
    y_pred = ensemble.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("── Confusion matrix (test set) ───────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())

    # ── Save ──────────────────────────────────────────────────
    print(f"\n── Saving to '{args.output}' ──────────────────────────")
    bundle = {
        "model":         ensemble,
        "label_encoder": le,
        "feature_count": X_train.shape[1],
        "classes":       list(le.classes_),
    }
    joblib.dump(bundle, args.output, compress=3)
    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"  Saved — {size_mb:.1f} MB")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
