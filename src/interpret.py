"""RQ4: what does each classifier look at?

1. TF-IDF + logistic regression: the n-grams with the largest positive
   weight for each family (the model is linear, so this is exact).
2. CNN: gradient x input saliency. For every correctly classified test
   response we take the gradient of the true-class logit w.r.t. each
   token's embedding, dot it with the embedding, and average that score
   per token type within each family.

    python -m src.interpret
"""
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.baselines import run_tfidf_lr
from src.common import FAMILIES, PROCESSED_PATH, RESULTS_DIR, RUNS_DIR, SEEDS, TABLE_DIR
from src.models import FamilyClassifier
from src.textproc import tokenize
from src.train import HP, MAX_LEN

TOP_K = 20
MIN_COUNT = 20  # a token must occur this often in a family's correct test examples


def lr_top_ngrams(tr, va, te):
    out_tmp = os.path.join(RUNS_DIR, "rq4_interpret", "tfidf_lr_output_refit.json")
    _, clf, vecs = run_tfidf_lr("output", tr, va, te, out_tmp, return_model=True, verbose=False)
    names = vecs["response"].get_feature_names_out()
    rows = []
    for ci, fam in enumerate(FAMILIES):
        w = clf.coef_[ci]
        for rank, j in enumerate(np.argsort(-w)[:TOP_K], 1):
            rows.append({"family": fam, "rank": rank, "ngram": names[j].replace("\n", "\\n"),
                         "weight": float(w[j])})
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "rq4_lr_top_ngrams.csv"), index=False)


def cnn_saliency(te, seed):
    stem = f"cnn_output_s{seed}"
    mdir = os.path.join(RESULTS_DIR, "models")
    with open(os.path.join(mdir, stem + "_vocab.pkl"), "rb") as fh:
        vocab = pickle.load(fh)
    hp = dict(HP["cnn"])
    hp.pop("lr")
    model = FamilyClassifier("cnn", ["response"], len(vocab), dropout=HP["common"]["dropout"], **hp)
    model.load_state_dict(torch.load(os.path.join(mdir, stem + ".pt")))
    model.eval()
    enc = model.encoders["response"]

    sums = {f: defaultdict(float) for f in FAMILIES}
    counts = {f: defaultdict(int) for f in FAMILIES}
    n_correct = {f: 0 for f in FAMILIES}
    for text, label in zip(te["response"].astype(str), te["label"]):
        toks = tokenize(text)[: MAX_LEN["response"]]
        ids = vocab.encode(toks, MAX_LEN["response"])
        T = max(len(ids), 5)
        x = torch.zeros((1, T), dtype=torch.long)
        x[0, :len(ids)] = torch.tensor(ids)
        lengths = torch.tensor([len(ids)])
        feats, emb = enc(x, lengths, return_emb=True)
        emb.retain_grad()
        logits = model.fc(feats)  # dropout is inactive in eval mode
        if int(logits.argmax()) != int(label):
            continue
        logits[0, int(label)].backward()
        score = (emb.grad[0] * emb[0]).sum(-1).detach().numpy()[:len(ids)]
        fam = FAMILIES[int(label)]
        n_correct[fam] += 1
        for tok_id, s in zip(ids, score):
            tok = vocab.itos[tok_id]
            sums[fam][tok] += float(s)
            counts[fam][tok] += 1
        model.zero_grad()

    rows = []
    for fam in FAMILIES:
        avg = {t: sums[fam][t] / counts[fam][t] for t in sums[fam]
               if counts[fam][t] >= MIN_COUNT and t not in ("<unk>",)}
        for rank, (tok, s) in enumerate(sorted(avg.items(), key=lambda kv: -kv[1])[:TOP_K], 1):
            rows.append({"family": fam, "rank": rank, "token": tok, "mean_saliency": s,
                         "occurrences": counts[fam][tok], "n_correct_examples": n_correct[fam]})
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "rq4_cnn_saliency_tokens.csv"), index=False)


def main():
    df = pd.read_csv(PROCESSED_PATH)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    lr_top_ngrams(tr, va, te)
    cnn_saliency(te, SEEDS[0])
    print("wrote rq4_lr_top_ngrams.csv and rq4_cnn_saliency_tokens.csv")


if __name__ == "__main__":
    main()
