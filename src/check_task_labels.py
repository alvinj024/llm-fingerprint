"""Draws a fixed random sample of prompts per task label so the keyword
rules in src/textproc.task_type can be checked by hand (RQ3 depends on them).

    python -m src.check_task_labels

Writes results/tables/task_label_audit.csv with an empty `correct` column:
fill it with 1/0 by hand, then run with --score to get per-task precision.
"""
import argparse
import os

import pandas as pd

from src.common import PROCESSED_PATH, TABLE_DIR
from src.textproc import TASKS

PER_TASK = 30
# Sample 0 was used to DEVELOP the rules; sample 1 (fresh prompts) is the
# held-out audit whose precision is reported.
AUDIT_SEED = 1
AUDIT = os.path.join(TABLE_DIR, "task_label_audit.csv")


def sample():
    df = pd.read_csv(PROCESSED_PATH).drop_duplicates("prompt_group")
    parts = [df[df.task == t].sample(n=min(PER_TASK, (df.task == t).sum()), random_state=AUDIT_SEED)
             for t in TASKS]
    out = pd.concat(parts)[["uid", "task", "prompt"]]
    out["prompt"] = out["prompt"].str.slice(0, 300).str.replace("\n", " ")
    out["correct"] = ""
    out.to_csv(AUDIT, index=False)
    print(f"wrote {AUDIT} ({len(out)} prompts)")


def score():
    a = pd.read_csv(AUDIT)
    a = a[a["correct"].astype(str).isin(["0", "1", "0.0", "1.0"])]
    a["correct"] = a["correct"].astype(float)
    res = a.groupby("task")["correct"].agg(["mean", "size"]).rename(
        columns={"mean": "precision", "size": "n_checked"})
    res.to_csv(os.path.join(TABLE_DIR, "task_label_precision.csv"))
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    score() if ap.parse_args().score else sample()
