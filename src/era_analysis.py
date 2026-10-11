"""RQ2 (iii): why is input-only classification slightly ABOVE chance even
though the Arena assigns models to prompts at random?

Hypothesis: temporal drift. Arena models are only sampled while they are
offered (e.g. Gemini Pro only from Dec 2023), and what users ask changes
over time (new topics, new jailbreak styles, asking about new models). So a
prompt carries information about WHEN it was written, and "when" is
correlated with WHICH models were available.

Test: within the two families whose versions span 2023 (GPT and Claude), we
label each example by its model's release era and ask whether the PROMPT
alone predicts the era. Because both eras contain the same family, any
signal here is time, not family identity.

    python -m src.era_analysis
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common import FAMILIES, PROCESSED_PATH, RUNS_DIR, TABLE_DIR, load_json, save_json
from src.textproc import tokenize

# Public release month of each model version (from the version suffix or the
# vendor's release announcement). "late" = released Nov 2023 or later.
RELEASE = {
    "gpt-4-0314": "2023-03", "gpt-3.5-turbo-0314": "2023-03",
    "claude-1": "2023-03", "claude-instant-1": "2023-03",
    "gpt-4-0613": "2023-06", "gpt-3.5-turbo-0613": "2023-06",
    "claude-2.0": "2023-07", "llama-2-7b-chat": "2023-07",
    "llama-2-13b-chat": "2023-07", "llama-2-70b-chat": "2023-07",
    "gpt-4-1106-preview": "2023-11", "gpt-3.5-turbo-1106": "2023-11",
    "claude-2.1": "2023-11",
    "gemini-pro": "2023-12", "gemini-pro-dev-api": "2023-12",
    "gpt-4-0125-preview": "2024-01", "gpt-3.5-turbo-0125": "2024-01",
}
LATE_FROM = "2023-11"
rng = np.random.default_rng(0)


def main():
    df = pd.read_csv(PROCESSED_PATH)
    missing = set(df.model) - set(RELEASE)
    assert not missing, f"no release date for {missing}"
    df["release"] = df.model.map(RELEASE)
    df["late"] = (df.release >= LATE_FROM).astype(int)
    era = pd.crosstab(df.family, df.late).rename(columns={0: "early (<Nov 2023)", 1: "late (>=Nov 2023)"})
    era.reindex(FAMILIES).to_csv(os.path.join(TABLE_DIR, "era_by_family.csv"))

    # ---- 1) within GPT + Claude: does the prompt predict the era?
    sub = df[df.family.isin(["GPT", "Claude"])]
    tr, va, te = (sub[sub.split == s] for s in ("train", "val", "test"))
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr, Xva, Xte = vec.fit_transform(tr.prompt), vec.transform(va.prompt), vec.transform(te.prompt)
    best = max(((roc_auc_score(va.late, LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
                               .fit(Xtr, tr.late).predict_proba(Xva)[:, 1]), C)
                for C in (0.1, 0.3, 1, 3, 10)))
    clf = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(Xtr, tr.late)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(te.late, p)
    boots = []
    y = te.late.to_numpy()
    for _ in range(2000):
        s = rng.choice(len(y), len(y), replace=True)
        if y[s].min() != y[s].max():
            boots.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # permutation null: shuffle era labels in training
    null = []
    for i in range(200):
        yp = rng.permutation(tr.late.to_numpy())
        null.append(roc_auc_score(y, LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced")
                                  .fit(Xtr, yp).predict_proba(Xte)[:, 1]))
    p_perm = (1 + sum(n >= auc for n in null)) / (1 + len(null))
    names = vec.get_feature_names_out()
    order = np.argsort(clf.coef_[0])
    out = {
        "families": ["GPT", "Claude"], "late_from": LATE_FROM,
        "n_train": len(tr), "n_test": len(te), "chosen_C": best[1],
        "test_auc": float(auc), "auc_95ci": [float(lo), float(hi)],
        "permutation_null_auc_mean": float(np.mean(null)), "permutation_p": float(p_perm),
        "top_late_ngrams": [names[j] for j in order[::-1][:20]],
        "top_early_ngrams": [names[j] for j in order[:20]],
    }

    # ---- 2) input-only family predictions: which families are "recognised"?
    lr = load_json(os.path.join(RUNS_DIR, "rq1_rq2", "tfidf_lr_input.json"))
    out["input_only_tfidf_per_family_recall"] = {f: lr["test"]["per_class"][f]["recall"] for f in FAMILIES}
    out["input_only_tfidf_confusion"] = lr["test"]["confusion"]
    save_json(out, os.path.join(RUNS_DIR, "rq2_era", "era_analysis.json"))
    print(era)
    print({k: v for k, v in out.items() if k not in ("top_late_ngrams", "top_early_ngrams")})
    print("late:", out["top_late_ngrams"][:12])
    print("early:", out["top_early_ngrams"][:12])


if __name__ == "__main__":
    main()
