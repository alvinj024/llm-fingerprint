"""Step 1: turn data/raw_4families.csv.gz into the cleaned, balanced and
split dataset data/processed.csv.gz, and write a data summary for the report.

    python -m src.data [--per_family N]

Pipeline (every step's counts are logged to results/tables/data_funnel.csv):
  1. keep examples whose prompt AND response are mostly English
  2. drop exact duplicate (family, prompt, response) triples
  3. label each prompt with a task type (coding / math / writing / general)
  4. balance: randomly sample exactly --per_family examples (default 1000)
     from every family, so the four classes are perfectly balanced
  5. split 70/15/15 into train/val/test, GROUPED BY PROMPT, so an identical
     prompt never appears in two splits (Arena has many repeated prompts and
     both sides of an Arena pair share their prompt)
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.common import (FAMILIES, LABEL2ID, PROCESSED_PATH, RAW_PATH,
                        TABLE_DIR, ensure_dirs, save_json)
from src.textproc import mostly_english, task_type

SPLIT_SEED = 0


def normalize_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


def clean(raw_path=RAW_PATH):
    """Steps 1-3 (English filter, de-duplication, task labels) on the whole
    raw file. Returns the cleaned frame and the funnel counts."""
    df = pd.read_csv(raw_path)
    df["prompt"] = df["prompt"].astype(str)
    df["response"] = df["response"].astype(str)
    funnel = [("raw single-turn examples in 4 families", len(df))]

    eng = df["prompt"].map(mostly_english) & df["response"].map(mostly_english)
    df = df[eng].copy()
    funnel.append(("after English filter", len(df)))

    df = df.drop_duplicates(subset=["family", "prompt", "response"]).copy()
    funnel.append(("after removing exact duplicates", len(df)))

    df["task"] = df["prompt"].map(task_type)
    df["prompt_group"] = df["prompt"].map(normalize_prompt)
    df["label"] = df["family"].map(LABEL2ID)
    return df, funnel


def grouped_split(df, seed=SPLIT_SEED):
    """70/15/15 train/val/test assignment, stratified by family and grouped
    by normalised prompt (20 folds: 0-2 test, 3-5 val, the rest train)."""
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    groups = pd.factorize(df["prompt_group"])[0]
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups)):
        fold[idx] = k
    return np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))


def build(per_family=None, raw_path=RAW_PATH, out_path=PROCESSED_PATH):
    ensure_dirs()
    df, funnel = clean(raw_path)
    before_balance = df["family"].value_counts().reindex(FAMILIES).fillna(0).astype(int)

    n = int(before_balance.min())
    if per_family:
        n = min(n, per_family)
    df = (df.groupby("family").sample(n=n, random_state=SPLIT_SEED)
            .reset_index(drop=True))
    funnel.append((f"after balancing ({n} per family)", len(df)))
    df["uid"] = np.arange(len(df))
    df["split"] = grouped_split(df)

    check_no_leakage(df)
    cols = ["uid", "pair_id", "side", "model", "family", "label", "task",
            "split", "prompt_group", "prompt", "response"]
    df[cols].to_csv(out_path, index=False, compression="gzip")

    # ---------------- summary tables for the report ----------------
    pd.DataFrame(funnel, columns=["step", "examples"]).to_csv(f"{TABLE_DIR}/data_funnel.csv", index=False)
    before_balance.rename("examples_before_balancing").to_csv(f"{TABLE_DIR}/family_counts_before_balancing.csv")
    pd.crosstab(df["family"], df["split"]).reindex(FAMILIES)[["train", "val", "test"]] \
        .to_csv(f"{TABLE_DIR}/split_counts.csv")
    pd.crosstab(df["family"], df["task"]).reindex(FAMILIES).to_csv(f"{TABLE_DIR}/family_by_task.csv")
    pd.crosstab(df["task"], df["split"]).to_csv(f"{TABLE_DIR}/task_by_split.csv")
    df.groupby(["family", "model"]).size().rename("examples").to_csv(f"{TABLE_DIR}/model_versions.csv")
    save_json({
        "per_family": n,
        "total": len(df),
        "n_prompt_groups": int(df["prompt_group"].nunique()),
        "funnel": funnel,
    }, f"{TABLE_DIR}/data_summary.json")

    print("\n".join(f"{s:45s} {c:>8,}" for s, c in funnel))
    print(pd.crosstab(df["family"], df["split"]).reindex(FAMILIES))
    return df


def check_no_leakage(df):
    """Fail loudly if any normalised prompt appears in more than one split."""
    spans = df.groupby("prompt_group")["split"].nunique()
    bad = int((spans > 1).sum())
    assert bad == 0, f"{bad} prompts appear in more than one split!"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_family", type=int, default=1000,
                    help="examples kept per family (project setting: 1000; 0 = smallest family size)")
    build(ap.parse_args().per_family)
