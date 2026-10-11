"""Training / evaluation for the CNN and BiLSTM classifiers.

Single run from the command line (normally you use run_experiments.py):
    python -m src.train --model cnn --view output --seed 13
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.common import (FAMILIES, PROCESSED_PATH, RESULTS_DIR, RUNS_DIR,
                        save_json, set_seed)
from src.models import FamilyClassifier
from src.textproc import Vocab, tokenize

torch.set_num_threads(max(1, os.cpu_count() or 1))

VIEWS = {"output": ["response"], "input": ["prompt"], "both": ["prompt", "response"]}
MAX_LEN = {"response": 512, "prompt": 256}

# Hyper-parameters (identical for every experiment; chosen once on the
# validation set of RQ1 and then frozen).
HP = {
    "cnn": dict(lr=1e-3, emb_dim=128, n_filters=128, kernel_sizes=(3, 4, 5)),
    "lstm": dict(lr=1e-3, emb_dim=128, hidden=128, layers=1),
    "common": dict(batch_size=64, max_epochs=30, patience=3, weight_decay=1e-4,
                   dropout=0.5, grad_clip=1.0, min_freq=2, max_vocab=30000),
}
if os.environ.get("SMOKE_TEST"):  # 1-epoch runs, only for checking the code paths
    HP["common"]["max_epochs"] = 1


# --------------------------------------------------------------------------
# Metrics (shared with the baselines)
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    labels = list(range(len(FAMILIES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {fam: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
                      for i, fam in enumerate(FAMILIES)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
def encode_split(df, fields, vocab, transform):
    enc = {}
    for f in fields:
        texts = df[f].astype(str)
        if transform is not None and f == "response":
            texts = texts.map(transform)
        enc[f] = [vocab.encode(tokenize(t), MAX_LEN[f]) for t in texts]
    enc["label"] = df["label"].to_numpy()
    enc["uid"] = df["uid"].to_numpy()
    return enc


def build_vocab(train_df, fields, transform, min_freq, max_vocab):
    lists = []
    for f in fields:
        texts = train_df[f].astype(str)
        if transform is not None and f == "response":
            texts = texts.map(transform)
        lists.extend(tokenize(t)[: MAX_LEN[f]] for t in texts)
    return Vocab(lists, min_freq=min_freq, max_size=max_vocab)


def batches(enc, fields, batch_size, shuffle, rng):
    """Length-bucketed batches: sort by length inside large chunks so that
    sequences in a batch have similar lengths (much faster on CPU), then
    shuffle the order of the batches."""
    n = len(enc["label"])
    key = np.array([len(enc[fields[-1]][i]) for i in range(n)])
    idx = rng.permutation(n) if shuffle else np.arange(n)
    chunk = batch_size * 50
    out = []
    for s in range(0, n, chunk):
        part = idx[s:s + chunk]
        part = part[np.argsort(key[part], kind="stable")]
        out.extend(part[i:i + batch_size] for i in range(0, len(part), batch_size))
    if shuffle:
        rng.shuffle(out)
    for b in out:
        yield collate(enc, fields, b)


def collate(enc, fields, ids):
    batch = {}
    for f in fields:
        seqs = [enc[f][i] for i in ids]
        lens = [len(s) for s in seqs]
        T = max(max(lens), 5)  # CNN needs at least the widest kernel
        arr = np.zeros((len(seqs), T), dtype=np.int64)
        for j, s in enumerate(seqs):
            arr[j, :len(s)] = s
        batch[f] = torch.from_numpy(arr)
        batch[f + "_len"] = torch.tensor(lens, dtype=torch.long)
    batch["label"] = torch.from_numpy(enc["label"][ids]).long()
    batch["uid"] = enc["uid"][ids]
    return batch


@torch.no_grad()
def predict(model, enc, fields, batch_size=128):
    model.eval()
    uids, probs = [], []
    for b in batches(enc, fields, batch_size, False, None):
        probs.append(torch.softmax(model(b), dim=1).numpy())
        uids.append(b["uid"])
    uids, probs = np.concatenate(uids), np.concatenate(probs)
    order = np.argsort(uids)  # restore a deterministic order
    return uids[order], probs[order]


# --------------------------------------------------------------------------
# One full training run
# --------------------------------------------------------------------------
def run(model_type, view, seed, train_df, val_df, test_df, out_path,
        transform=None, save_model=False, verbose=True):
    stem = os.path.splitext(os.path.basename(out_path))[0]
    weights = os.path.join(RESULTS_DIR, "models", stem + ".pt")
    # Skip finished runs - unless this run's weights are needed (save_model)
    # and missing, e.g. in a fresh clone (weights are not committed).
    if os.path.exists(out_path) and (not save_model or os.path.exists(weights)):
        if verbose:
            print(f"skip (exists): {out_path}")
        return None
    set_seed(seed)
    fields = VIEWS[view]
    c = HP["common"]
    hp = dict(HP[model_type])
    lr = hp.pop("lr")
    vocab = build_vocab(train_df, fields, transform, c["min_freq"], c["max_vocab"])
    tr = encode_split(train_df, fields, vocab, transform)
    va = encode_split(val_df, fields, vocab, transform)
    te = encode_split(test_df, fields, vocab, transform)

    model = FamilyClassifier(model_type, fields, len(vocab), dropout=c["dropout"], **hp)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    best_f1, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, c["max_epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for b in batches(tr, fields, c["batch_size"], True, rng):
            opt.zero_grad()
            loss = loss_fn(model(b), b["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            opt.step()
            tot += loss.item() * len(b["label"])
            n += len(b["label"])
        _, vp = predict(model, va, fields)
        v_true = va["label"][np.argsort(va["uid"])]
        v_pred = vp.argmax(1)
        v_f1 = f1_score(v_true, v_pred, average="macro")
        v_acc = accuracy_score(v_true, v_pred)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_macro_f1": float(v_f1),
                        "val_accuracy": float(v_acc)})
        if verbose:
            print(f"  ep {epoch:2d} loss {tot / n:.4f} val_f1 {v_f1:.4f} val_acc {v_acc:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if v_f1 > best_f1:
            best_f1, best_epoch, bad = v_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    uids, probs = predict(model, te, fields)
    y_true = te["label"][np.argsort(te["uid"])]
    y_pred = probs.argmax(1)
    res = {
        "model": model_type, "view": view, "seed": seed,
        "transform": getattr(transform, "__name__", None) if transform else None,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "vocab_size": len(vocab), "n_params": int(n_params),
        "hyperparameters": {**HP["common"], **HP[model_type],
                            "max_len": {f: MAX_LEN[f] for f in fields}},
        "best_epoch": best_epoch, "best_val_macro_f1": float(best_f1),
        "train_time_sec": round(train_time, 1), "history": history,
        "test": compute_metrics(y_true, y_pred),
        "test_predictions": {"uid": uids.tolist(), "true": y_true.tolist(),
                             "pred": y_pred.tolist(),
                             "probs": np.round(probs, 4).tolist()},
    }
    save_json(res, out_path)
    if save_model:
        mdir = os.path.join(RESULTS_DIR, "models")
        os.makedirs(mdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(out_path))[0]
        torch.save(model.state_dict(), os.path.join(mdir, stem + ".pt"))
        with open(os.path.join(mdir, stem + "_vocab.pkl"), "wb") as fh:
            pickle.dump(vocab, fh)
    if verbose:
        m = res["test"]
        print(f"TEST {model_type}/{view}/seed{seed}: acc {m['accuracy']:.4f} "
              f"macro-F1 {m['macro_f1']:.4f}  (best epoch {best_epoch}, {train_time:.0f}s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--view", choices=list(VIEWS), default="output")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    df = pd.read_csv(PROCESSED_PATH)
    run(a.model, a.view, a.seed, df[df.split == "train"], df[df.split == "val"],
        df[df.split == "test"], os.path.join(RUNS_DIR, "manual", f"{a.model}_{a.view}_s{a.seed}.json"))
