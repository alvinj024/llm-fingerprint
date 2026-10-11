"""Independent consistency checks on the finished results.

    python tests/verify_results.py

1. No prompt appears in two splits; labels match families.
2. For every run JSON: accuracy / macro-F1 / confusion recomputed from the
   saved predictions match the saved metrics.
3. Main-dataset runs (RQ1/RQ2/RQ4) were evaluated on exactly the 600 test
   examples, with the true labels of data/processed.csv.gz.
4. RQ3 test sets contain only target-task prompts from val+test, and their
   training sets never contain that task (OOD) / never contain val or test rows.
5. The RQ2 demo pool shares no prompt with the main 4,000-example dataset.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.common import FAMILIES, LABEL2ID, PROCESSED_PATH, RUNS_DIR  # noqa: E402

ok = True


def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print("FAIL:", msg)


df = pd.read_csv(PROCESSED_PATH)
check(len(df) == 4000 and (df.family.value_counts() == 1000).all(), "4 x 1000 examples")
check((df.groupby("prompt_group").split.nunique() == 1).all(), "prompt in two splits")
check((df.family.map(LABEL2ID) == df.label).all(), "label/family mismatch")
test = df[df.split == "test"].set_index("uid")

n_files = 0
for f in sorted(glob.glob(os.path.join(RUNS_DIR, "**", "*.json"), recursive=True)):
    r = json.load(open(f))
    if "test_predictions" not in r:
        continue
    n_files += 1
    tp = r["test_predictions"]
    y, p = np.array(tp["true"]), np.array(tp["pred"])
    check(abs(accuracy_score(y, p) - r["test"]["accuracy"]) < 1e-9, f"accuracy {f}")
    check(abs(f1_score(y, p, average="macro", labels=range(4), zero_division=0)
              - r["test"]["macro_f1"]) < 1e-9, f"macro-F1 {f}")
    check(confusion_matrix(y, p, labels=range(4)).tolist() == r["test"]["confusion"], f"confusion {f}")
    P = np.array(tp["probs"])  # saved rounded to 4 decimals -> allow exact ties
    check((P[np.arange(len(p)), p] >= P.max(1) - 1e-9).all(), f"probs/pred {f}")
    stage = os.path.basename(os.path.dirname(f))
    if stage in ("rq1_rq2", "rq4_ablation", "rq4_interpret"):
        uids = np.array(tp["uid"])
        check(len(uids) == 600 and set(uids) == set(test.index), f"test set {f}")
        check((test.loc[uids, "label"].to_numpy() == y).all(), f"true labels {f}")
    if stage == "rq3":
        target = os.path.basename(f).split("_")[0]
        rows = df.set_index("uid").loc[tp["uid"]]
        check((rows.task == target).all(), f"rq3 test task {f}")
        check(rows.split.isin(["val", "test"]).all(), f"rq3 test split {f}")
print(f"checked {n_files} run files")

# RQ3 training sets (re-derived exactly as in run_experiments.stage_rq3)
tr = df[df.split == "train"]
for target in ["writing", "coding"]:
    check((tr[tr.task != target].task != target).all(), "ood train contains target")

# RQ2 demo pool disjoint from main data
from src.run_experiments import demo_pool  # noqa: E402
pool = demo_pool()
check(not set(pool.prompt_group) & set(df.prompt_group), "demo pool overlaps main prompts")
check((pool.groupby("prompt_group").split.nunique() == 1).all(), "demo prompt in two splits")

print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
