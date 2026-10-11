"""Runs every experiment in the report. Resumable: finished runs (result
JSON already on disk) are skipped, so you can stop and restart freely.

    python -m src.run_experiments --stage rq1_rq2
    python -m src.run_experiments --stage rq2_confounded
    python -m src.run_experiments --stage rq3
    python -m src.run_experiments --stage rq4
    python -m src.run_experiments --stage all

Results land in results/runs/<stage>/*.json (one file per run).
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.baselines import run_majority, run_tfidf_lr
from src.common import FAMILIES, PROCESSED_PATH, RUNS_DIR, SEEDS, TABLE_DIR, save_json
from src.textproc import ABLATIONS
from src.train import run


def load():
    df = pd.read_csv(PROCESSED_PATH)
    return df, df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def lstm_seeds(n):
    return SEEDS[:n]


# --------------------------------------------------------------------------
# RQ1 + RQ2: output only / input only / input + output
# --------------------------------------------------------------------------
def stage_rq1_rq2(n_lstm_seeds):
    _, tr, va, te = load()
    d = os.path.join(RUNS_DIR, "rq1_rq2")
    run_majority(tr, te, os.path.join(d, "majority.json"))
    for view in ["output", "input", "both"]:
        run_tfidf_lr(view, tr, va, te, os.path.join(d, f"tfidf_lr_{view}.json"))
    for view in ["output", "input", "both"]:
        for seed in SEEDS:
            run("cnn", view, seed, tr, va, te, os.path.join(d, f"cnn_{view}_s{seed}.json"),
                save_model=(view == "output"))
        for seed in lstm_seeds(n_lstm_seeds):
            run("lstm", view, seed, tr, va, te, os.path.join(d, f"lstm_{view}_s{seed}.json"),
                save_model=(view == "output"))


# --------------------------------------------------------------------------
# RQ2 contrast: a simulated CONFOUNDED collection. Each family's prompts are
# drawn mostly from one task type, as can happen when each model's responses
# are collected from a different source. This shows how input-only
# classification can look "surprisingly good" because of how data was
# collected rather than because the prompt reveals the model.
#
# The 1,000-per-family main dataset is too small for this, so the demo uses
# ONLY the cleaned Arena responses the main dataset did not sample, and drops
# every prompt that occurs in the main dataset. It has its own grouped split.
# Control condition: the same pool and sizes, but every family gets the SAME
# (pool-wide) task mix -> isolates the effect of the confound.
# --------------------------------------------------------------------------
DEMO_TASKS = ["coding", "writing", "general"]  # math is too rare to skew on
CONFOUND_MIX = {  # share of each family's examples per task
    "GPT":    {"coding": 0.70, "writing": 0.15, "general": 0.15},
    "Claude": {"coding": 0.15, "writing": 0.70, "general": 0.15},
    "Gemini": {"coding": 0.15, "writing": 0.15, "general": 0.70},
    "Llama":  {"coding": 1 / 3, "writing": 1 / 3, "general": 1 / 3},
}


def demo_pool():
    from src.data import clean, grouped_split
    main = pd.read_csv(PROCESSED_PATH)
    pool, _ = clean()
    pool = pool[~pool.prompt_group.isin(set(main.prompt_group))]
    pool = pool[pool.task.isin(DEMO_TASKS)].reset_index(drop=True)
    pool["uid"] = 100_000 + np.arange(len(pool))  # never collides with main uids
    pool["split"] = grouped_split(pool)
    return pool


def mixture_size(split_df, mix):
    """Largest per-family size n such that family f can supply n*mix[f][t]
    examples of every task t."""
    return min(int((split_df[(split_df.family == f)].task == t).sum() / share)
               for f in FAMILIES for t, share in mix[f].items() if share > 0)


def mixture_subset(split_df, mix, n, seed=0):
    """Per-family sample of size n with task proportions mix[f]."""
    parts = []
    for f in FAMILIES:
        g = split_df[split_df.family == f]
        for t, share in mix[f].items():
            parts.append(g[g.task == t].sample(n=int(round(n * share)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def stage_rq2_confounded():
    pool = demo_pool()
    natural = (pool.task.value_counts(normalize=True)).to_dict()
    random_mix = {f: natural for f in FAMILIES}
    d = os.path.join(RUNS_DIR, "rq2_confounded")
    design = {"pool_size": len(pool), "pool_by_family": pool.family.value_counts().to_dict(),
              "confounded_mix": CONFOUND_MIX, "random_mix": natural}
    for cond, mix in [("random", random_mix), ("confounded", CONFOUND_MIX)]:
        # identical per-family sizes in both conditions (the smaller of the two)
        tr, va, te = (mixture_subset(pool[pool.split == s], mix,
                                     min(mixture_size(pool[pool.split == s], random_mix),
                                         mixture_size(pool[pool.split == s], CONFOUND_MIX)))
                      for s in ("train", "val", "test"))
        design[cond] = {"sizes": {"train": len(tr), "val": len(va), "test": len(te)},
                        "train_family_by_task": pd.crosstab(tr.family, tr.task).to_dict()}
        for view in ["input", "output"]:
            run_tfidf_lr(view, tr, va, te, os.path.join(d, f"{cond}_tfidf_lr_{view}.json"))
            for seed in SEEDS:
                run("cnn", view, seed, tr, va, te, os.path.join(d, f"{cond}_cnn_{view}_s{seed}.json"))
    save_json(design, os.path.join(d, "design.json"))


# --------------------------------------------------------------------------
# RQ3: train on some task types, test on an unseen one
# --------------------------------------------------------------------------
def stage_rq3(n_lstm_seeds):
    _, tr, va, te = load()
    d = os.path.join(RUNS_DIR, "rq3")
    sizes = {}
    for target in ["writing", "coding"]:
        tr_ood, va_ood = tr[tr.task != target], va[va.task != target]
        # Neither condition below trains on, or early-stops on, the target-task
        # prompts of the val split (early stopping uses va_ood, which excludes
        # them; training uses the train split only). So the target-task rows of
        # BOTH val and test are unseen held-out data for both conditions, and
        # we test on all of them to double the RQ3 test set. Prompts never
        # cross splits, so this does not leak.
        te_target = pd.concat([va[va.task == target], te[te.task == target]])
        # Control: same number of training examples, but drawn from ALL tasks
        # (so the target task is seen in training). Validation set is the
        # same for both conditions so model selection is comparable.
        tr_id = tr.sample(n=len(tr_ood), random_state=0)
        sizes[target] = {"train": len(tr_ood), "val": len(va_ood), "test_target": len(te_target),
                         "test_target_by_family": te_target.family.value_counts().to_dict()}
        for cond, trn in [("ood", tr_ood), ("id", tr_id)]:
            run_tfidf_lr("output", trn, va_ood, te_target,
                         os.path.join(d, f"{target}_{cond}_tfidf_lr.json"))
            for seed in SEEDS:
                run("cnn", "output", seed, trn, va_ood, te_target,
                    os.path.join(d, f"{target}_{cond}_cnn_s{seed}.json"))
            for seed in lstm_seeds(n_lstm_seeds):
                run("lstm", "output", seed, trn, va_ood, te_target,
                    os.path.join(d, f"{target}_{cond}_lstm_s{seed}.json"))
    save_json(sizes, os.path.join(d, "design.json"))


# --------------------------------------------------------------------------
# RQ4: ablations - remove a signal from the text, retrain, re-test
# --------------------------------------------------------------------------
def stage_rq4(n_lstm_seeds):
    _, tr, va, te = load()
    d = os.path.join(RUNS_DIR, "rq4_ablation")
    for name, fn in ABLATIONS.items():
        if name == "none":
            continue  # the unablated numbers are the RQ1 output-only runs
        fn.__name__ = name
        run_tfidf_lr("output", tr, va, te, os.path.join(d, f"{name}_tfidf_lr.json"), transform=fn)
        for seed in SEEDS:
            run("cnn", "output", seed, tr, va, te, os.path.join(d, f"{name}_cnn_s{seed}.json"),
                transform=fn)
        for seed in lstm_seeds(n_lstm_seeds):
            run("lstm", "output", seed, tr, va, te, os.path.join(d, f"{name}_lstm_s{seed}.json"),
                transform=fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["rq1_rq2", "rq2_confounded", "rq3", "rq4", "all"])
    ap.add_argument("--lstm_seeds", type=int, default=3,
                    help="seeds for the (slow) BiLSTM in RQ3/RQ4; RQ1/RQ2 always use this too")
    a = ap.parse_args()
    if a.stage in ("rq1_rq2", "all"):
        stage_rq1_rq2(a.lstm_seeds)
    if a.stage in ("rq2_confounded", "all"):
        stage_rq2_confounded()
    if a.stage in ("rq3", "all"):
        stage_rq3(a.lstm_seeds)
    if a.stage in ("rq4", "all"):
        stage_rq4(a.lstm_seeds)
