"""RQ4: hand-crafted linguistic / structural features of each response, how
they differ by family, and how well each feature GROUP alone identifies
the family.

    python -m src.stylometry

Writes results/tables/stylometry_*.csv and results/runs/rq4_stylometry/*.json
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from src.common import FAMILIES, PROCESSED_PATH, RUNS_DIR, TABLE_DIR, save_json
from src.train import compute_metrics

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

# Opening / closing / refusal phrases checked as binary flags.
PHRASES = {
    "opens_sure_certainly": r"^\W*(sure|certainly|of course|absolutely)\b",
    "opens_greeting": r"^\W*(hello|hi|hey|greetings)\b",
    "opens_apology_refusal": r"^\W*(i'?m sorry|i am sorry|i apologi[sz]e|i cannot|i can'?t|i'?m not able|i am not able|i'?m unable)",
    "mentions_as_an_ai": r"\bas an ai\b|\bas a language model\b|\bai language model\b",
    "closes_offer_help": r"(let me know|feel free|hope this helps|i hope this|happy to help)[^\n]{0,80}\W*$",
    "has_in_summary": r"\b(in summary|in conclusion|overall,|to summarize|ultimately,)",
}
PHRASE_RE = {k: re.compile(v, re.I) for k, v in PHRASES.items()}


def mtld(words, threshold=0.72):
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010),
    averaged over a forward and a backward pass. Largely independent of
    text length, unlike a plain type-token ratio."""
    def one_pass(ws):
        factors, types, count = 0.0, set(), 0
        for w in ws:
            types.add(w)
            count += 1
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if ttr < 1 else 0
        return len(ws) / factors if factors > 0 else np.nan
    if len(words) < 50:
        return np.nan
    return float(np.nanmean([one_pass(words), one_pass(words[::-1])]))


def features(text: str) -> dict:
    words = WORD_RE.findall(text)
    lw = [w.lower() for w in words]
    nw = max(len(words), 1)
    lines = [l for l in text.split("\n") if l.strip()]
    nl = max(len(lines), 1)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if WORD_RE.search(s)]
    f = {
        # ---- length
        "n_words": len(words),
        "n_chars": len(text),
        "n_lines": len(lines),
        "n_paragraphs": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
        # ---- lexical
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_sentence_words": len(words) / max(len(sentences), 1),
        "ttr_first100": len(set(lw[:100])) / max(len(lw[:100]), 1),
        "mtld": mtld(lw),
        "first_person_I_per100": 100 * sum(1 for w in words if w == "I") / nw,
        "you_per100": 100 * sum(1 for w in lw if w in ("you", "your")) / nw,
        # ---- formatting / structure
        "headers": sum(1 for l in lines if re.match(r"^\s*#{1,6}\s", l)),
        "bullet_lines": sum(1 for l in lines if re.match(r"^\s*[-*•+]\s", l)),
        "numbered_lines": sum(1 for l in lines if re.match(r"^\s*\d+[.)]\s", l)),
        "bold_spans": text.count("**") // 2,
        "code_blocks": text.count("```") // 2,
        "list_line_frac": sum(1 for l in lines if re.match(r"^\s*([-*•+]|\d+[.)])\s", l)) / nl,
        # ---- punctuation / tone
        "exclaim_per100": 100 * text.count("!") / nw,
        "question_per100": 100 * text.count("?") / nw,
        "colon_per100": 100 * text.count(":") / nw,
        "emdash_per100": 100 * (text.count("—") + text.count(" - ")) / nw,
        "emoji_count": len(EMOJI_RE.findall(text)),
    }
    for k, rx in PHRASE_RE.items():
        f[k] = int(bool(rx.search(text.strip())))
    return f


GROUPS = {
    "length": ["n_words", "n_chars", "n_lines", "n_paragraphs"],
    "lexical": ["mean_word_len", "mean_sentence_words", "ttr_first100", "mtld",
                "first_person_I_per100", "you_per100"],
    "formatting": ["headers", "bullet_lines", "numbered_lines", "bold_spans",
                   "code_blocks", "list_line_frac"],
    "punctuation_tone": ["exclaim_per100", "question_per100", "colon_per100",
                         "emdash_per100", "emoji_count"],
    "stock_phrases": list(PHRASES),
}
GROUPS["all_features"] = [f for g in GROUPS.values() for f in g]


def main():
    df = pd.read_csv(PROCESSED_PATH)
    F = pd.DataFrame([features(t) for t in df["response"].astype(str)])
    F["family"], F["split"], F["label"] = df["family"].values, df["split"].values, df["label"].values
    F.to_csv(os.path.join(TABLE_DIR, "stylometry_features_all.csv.gz"), index=False, compression="gzip")

    # ---- per-family descriptive statistics (all splits) + Kruskal-Wallis
    rows = []
    for col in GROUPS["all_features"]:
        r = {"feature": col}
        for fam in FAMILIES:
            v = F.loc[F.family == fam, col].dropna()
            r[f"{fam}_mean"] = v.mean()
            r[f"{fam}_median"] = v.median()
        samples = [F.loc[F.family == fam, col].dropna() for fam in FAMILIES]
        try:
            stat, p = kruskal(*samples)
        except ValueError:  # all values identical
            stat, p = 0.0, 1.0
        if not np.isfinite(stat):
            stat, p = 0.0, 1.0
        r["kruskal_H"], r["p_value"] = stat, p
        n = sum(len(s) for s in samples)
        # epsilon-squared effect size for Kruskal-Wallis
        r["epsilon_sq"] = stat * (n + 1) / (n ** 2 - 1)
        rows.append(r)
    desc = pd.DataFrame(rows).sort_values("epsilon_sq", ascending=False)
    desc.to_csv(os.path.join(TABLE_DIR, "stylometry_by_family.csv"), index=False)

    # ---- which feature groups alone can identify the family?
    tr, va, te = (F[F.split == s] for s in ("train", "val", "test"))
    out = {}
    for g, cols in GROUPS.items():
        best = None
        for lr in (0.05, 0.1):
            for leaf in (15, 31):
                clf = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaf,
                                                     max_iter=300, random_state=0)
                clf.fit(tr[cols], tr["label"])
                v = f1_score(va["label"], clf.predict(va[cols]), average="macro")
                if best is None or v > best[0]:
                    best = (v, clf, {"learning_rate": lr, "max_leaf_nodes": leaf})
        v, clf, params = best
        pred = clf.predict(te[cols])
        out[g] = {"features": cols, "params": params, "val_macro_f1": float(v),
                  "test": compute_metrics(te["label"].to_numpy(), pred)}
        print(f"{g:18s} test acc {out[g]['test']['accuracy']:.4f} "
              f"macro-F1 {out[g]['test']['macro_f1']:.4f}")
    save_json(out, os.path.join(RUNS_DIR, "rq4_stylometry", "feature_groups.json"))


if __name__ == "__main__":
    main()
