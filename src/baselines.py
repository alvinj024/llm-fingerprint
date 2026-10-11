"""Non-neural reference points: majority class and TF-IDF + logistic
regression. Used in RQ1/RQ2 (as baselines), RQ3, and RQ4 (as an
interpretable model whose weights we can read)."""
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.common import FAMILIES, save_json
from src.textproc import tokenize
from src.train import VIEWS, compute_metrics

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def _vectorizer():
    # Same tokenizer as the neural models (case, punctuation, newlines and
    # markdown kept); unigrams + bigrams.
    return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                           ngram_range=(1, 2), min_df=2, max_features=200_000,
                           sublinear_tf=True)


def _features(train_df, other_dfs, fields, transform):
    mats_tr, mats_other, vecs = [], [[] for _ in other_dfs], {}
    for f in fields:
        def col(d):
            s = d[f].astype(str)
            return s.map(transform) if (transform is not None and f == "response") else s
        v = _vectorizer()
        mats_tr.append(v.fit_transform(col(train_df)))
        for j, d in enumerate(other_dfs):
            mats_other[j].append(v.transform(col(d)))
        vecs[f] = v
    return sp.hstack(mats_tr).tocsr(), [sp.hstack(m).tocsr() for m in mats_other], vecs


def run_tfidf_lr(view, train_df, val_df, test_df, out_path, transform=None,
                 return_model=False, verbose=True):
    """Fits TF-IDF+LR, picks C on the validation set, evaluates on test."""
    if os.path.exists(out_path) and not return_model:
        if verbose:
            print(f"skip (exists): {out_path}")
        return None
    fields = VIEWS[view]
    Xtr, (Xva, Xte), vecs = _features(train_df, [val_df, test_df], fields, transform)
    ytr, yva = train_df["label"].to_numpy(), val_df["label"].to_numpy()
    best = (-1, None, None)
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        if f1 > best[0]:
            best = (f1, C, clf)
    val_f1, C, clf = best
    probs = clf.predict_proba(Xte)
    y_true, y_pred = test_df["label"].to_numpy(), probs.argmax(1)
    res = {
        "model": "tfidf_lr", "view": view, "seed": None,
        "transform": getattr(transform, "__name__", None) if transform else None,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "chosen_C": C, "best_val_macro_f1": float(val_f1),
        "n_features": int(Xtr.shape[1]),
        "test": compute_metrics(y_true, y_pred),
        "test_predictions": {"uid": test_df["uid"].tolist(), "true": y_true.tolist(),
                             "pred": y_pred.tolist(), "probs": np.round(probs, 4).tolist()},
    }
    save_json(res, out_path)
    if verbose:
        print(f"TEST tfidf_lr/{view}: acc {res['test']['accuracy']:.4f} "
              f"macro-F1 {res['test']['macro_f1']:.4f} (C={C})")
    return (res, clf, vecs) if return_model else res


def run_majority(train_df, test_df, out_path):
    """Predicts the most frequent training label. With balanced classes this
    is chance level (25% accuracy)."""
    maj = int(train_df["label"].mode()[0])
    y_true = test_df["label"].to_numpy()
    y_pred = np.full_like(y_true, maj)
    res = {"model": "majority", "view": "none", "seed": None,
           "majority_class": FAMILIES[maj], "n_test": len(test_df),
           "test": compute_metrics(y_true, y_pred)}
    save_json(res, out_path)
    return res
