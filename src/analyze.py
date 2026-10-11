"""Collects every run's JSON into the tables and figures used in the report.

    python -m src.analyze

Outputs: results/tables/*.csv|*.md, results/figures/*.png and
results/SUMMARY.md (every headline number in one place).
"""
import glob
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from src.common import (FAMILIES, FIG_DIR, PROCESSED_PATH, RUNS_DIR, SEEDS,
                        TABLE_DIR, load_json)

# ---- figure style (validated categorical palette, recessive axes) --------
FAM_COLOR = {"GPT": "#2a78d6", "Claude": "#eb6834", "Gemini": "#1baf7a", "Llama": "#4a3aa7"}
MODEL_COLOR = {"tfidf_lr": "#2a78d6", "cnn": "#eb6834", "lstm": "#1baf7a"}
MODEL_NAME = {"tfidf_lr": "TF-IDF + LR", "cnn": "CNN", "lstm": "BiLSTM", "majority": "Majority"}
VIEW_NAME = {"output": "Output only", "input": "Input only", "both": "Input + Output"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "legend.frameon": False,
})
CHANCE = 1 / len(FAMILIES)
rng = np.random.default_rng(0)
SUMMARY = []


def note(line=""):
    SUMMARY.append(line)


def runs(stage, pattern):
    files = sorted(glob.glob(os.path.join(RUNS_DIR, stage, pattern)))
    return [load_json(f) for f in files]


def mstd(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 1:
        return f"{100 * vals[0]:.1f}"
    return f"{100 * vals.mean():.1f} ± {100 * vals.std(ddof=1):.1f}"


def ensemble(rs):
    """Average the predicted probabilities of several seeds of one config."""
    P = np.mean([np.array(r["test_predictions"]["probs"]) for r in rs], axis=0)
    first = rs[0]["test_predictions"]
    for r in rs[1:]:
        assert r["test_predictions"]["uid"] == first["uid"]
    return np.array(first["uid"]), np.array(first["true"]), P.argmax(1)


def bootstrap_ci(y, p, metric, n=2000):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        vals.append(metric(y[s], p[s]))
    return np.percentile(vals, [2.5, 97.5])


def macro(y, p):
    return f1_score(y, p, average="macro", labels=list(range(len(FAMILIES))), zero_division=0)


def to_md(df, path, floatfmt=None):
    df.to_csv(path.replace(".md", ".csv"), index=False)
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False))


def bar_labels(ax, bars, vals):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{100 * v:.1f}",
                ha="center", va="bottom", fontsize=8, color=INK)


# ==========================================================================
def data_section():
    df = pd.read_csv(PROCESSED_PATH)
    note("## Data")
    for f in ("data_funnel", "split_counts", "family_by_task", "model_versions"):
        p = os.path.join(TABLE_DIR, f + ".csv")
        note(f"### {f}\n" + pd.read_csv(p).to_markdown(index=False) + "\n")
    df["n_words"] = df["response"].astype(str).str.split().str.len()
    df["p_words"] = df["prompt"].astype(str).str.split().str.len()
    from src.textproc import tokenize
    df["n_tok"] = df["response"].astype(str).map(lambda t: len(tokenize(t)))
    df["p_tok"] = df["prompt"].astype(str).map(lambda t: len(tokenize(t)))
    note(f"Responses truncated at 512 tokens: {100 * (df.n_tok > 512).mean():.1f}% ; "
         f"prompts truncated at 256 tokens: {100 * (df.p_tok > 256).mean():.1f}%")
    note(f"Median response tokens {df.n_tok.median():.0f}, median prompt tokens {df.p_tok.median():.0f}\n")
    return df


# ==========================================================================
def rq1_rq2(df):
    note("## RQ1 / RQ2 (random-assignment Arena data)")
    rows, ens_preds = [], {}
    maj = runs("rq1_rq2", "majority.json")[0]
    rows.append({"model": "Majority", "view": "-", "seeds": 1,
                 "accuracy": mstd([maj["test"]["accuracy"]]),
                 "macro_f1": mstd([maj["test"]["macro_f1"]]), "_f1": maj["test"]["macro_f1"],
                 "_std": 0})
    for view in ["output", "input", "both"]:
        for m in ["tfidf_lr", "cnn", "lstm"]:
            rs = runs("rq1_rq2", f"{m}_{view}.json" if m == "tfidf_lr" else f"{m}_{view}_s*.json")
            if not rs:
                continue
            acc = [r["test"]["accuracy"] for r in rs]
            f1 = [r["test"]["macro_f1"] for r in rs]
            uid, y, p = ensemble(rs)
            ens_preds[(m, view)] = (uid, y, p)
            lo, hi = bootstrap_ci(y, p, macro)
            rows.append({"model": MODEL_NAME[m], "view": VIEW_NAME[view], "seeds": len(rs),
                         "accuracy": mstd(acc), "macro_f1": mstd(f1),
                         "ensemble_macro_f1_95ci": f"{100 * macro(y, p):.1f} [{100 * lo:.1f}, {100 * hi:.1f}]",
                         "best_epoch": ",".join(str(r.get("best_epoch", "-")) for r in rs),
                         "train_min": ",".join(f"{r.get('train_time_sec', 0) / 60:.1f}" for r in rs),
                         "_f1": np.mean(f1), "_std": np.std(f1, ddof=1) if len(f1) > 1 else 0,
                         "_m": m, "_view": view})
    tab = pd.DataFrame(rows)
    to_md(tab[[c for c in tab.columns if not c.startswith("_")]].fillna("-"),
          os.path.join(TABLE_DIR, "rq1_rq2_main.md"))
    note(open(os.path.join(TABLE_DIR, "rq1_rq2_main.md")).read() + "\n")

    # ---- significance: input-only vs chance; CNN vs BiLSTM on output
    for m in ["tfidf_lr", "cnn", "lstm"]:
        if (m, "input") in ens_preds:
            _, y, p = ens_preds[(m, "input")]
            k = int((y == p).sum())
            bt = binomtest(k, len(y), CHANCE, alternative="greater")
            note(f"Input-only {MODEL_NAME[m]} (seed ensemble): acc {100 * k / len(y):.2f}% "
                 f"({k}/{len(y)}), one-sided binomial test vs 25%: p = {bt.pvalue:.3g}")
    if ("cnn", "output") in ens_preds and ("lstm", "output") in ens_preds:
        _, y, pc = ens_preds[("cnn", "output")]
        _, _, pl = ens_preds[("lstm", "output")]
        b = int(((pc == y) & (pl != y)).sum())
        c = int(((pc != y) & (pl == y)).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        note(f"McNemar (exact) CNN vs BiLSTM, output only, seed ensembles: CNN-only-correct {b}, "
             f"BiLSTM-only-correct {c}, p = {p:.3g}")
    for (m, view), (_, y, p) in sorted(ens_preds.items()):
        if view == "output":
            pass
    note("")

    # ---- figure: macro-F1 by view and model
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    views = ["output", "input", "both"]
    ms = ["tfidf_lr", "cnn", "lstm"]
    w = 0.26
    t = tab[tab.model != "Majority"]
    for i, m in enumerate(ms):
        sub = [t[(t._m == m) & (t._view == v)] for v in views]
        vals = [s._f1.iloc[0] if len(s) else np.nan for s in sub]
        errs = [s._std.iloc[0] if len(s) else 0 for s in sub]
        x = np.arange(len(views)) + (i - 1) * w
        bars = ax.bar(x, vals, w - 0.03, yerr=errs, color=MODEL_COLOR[m], label=MODEL_NAME[m],
                      error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
        bar_labels(ax, bars, vals)
    ax.axhline(CHANCE, ls="--", lw=1, color=MUTED, label="Chance (25%)")
    ax.set_xticks(range(len(views)), [VIEW_NAME[v] for v in views])
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    ax.set_title("Which text identifies the LLM? (mean ± std over seeds)", fontsize=10, loc="left")
    fig.savefig(os.path.join(FIG_DIR, "fig_rq1_rq2_views.png"))
    plt.close(fig)

    # ---- per-family F1 (output only)
    rows = []
    for m in ms:
        rs = runs("rq1_rq2", f"{m}_output.json" if m == "tfidf_lr" else f"{m}_output_s*.json")
        if not rs:
            continue
        r = {"model": MODEL_NAME[m]}
        for fam in FAMILIES:
            for k in ("precision", "recall", "f1"):
                r[f"{fam}_{k}"] = mstd([x["test"]["per_class"][fam][k] for x in rs])
        rows.append(r)
    pf = pd.DataFrame(rows)
    to_md(pf, os.path.join(TABLE_DIR, "rq1_per_family.md"))
    note("### Per-family precision/recall/F1 (output only)\n" + pf.to_markdown(index=False) + "\n")

    # ---- confusion matrices (seed ensembles)
    avail = [m for m in ("cnn", "lstm") if (m, "output") in ens_preds]
    fig, axes = plt.subplots(1, len(avail), figsize=(3.4 * len(avail), 3.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, avail):
        _, y, p = ens_preds[(m, "output")]
        cm = pd.crosstab(pd.Categorical(y, categories=range(4)),
                         pd.Categorical(p, categories=range(4)), dropna=False).to_numpy()
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100 * cmn[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center",
                        fontsize=7.5, color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), FAMILIES)
        ax.set_yticks(range(4), FAMILIES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(f"{MODEL_NAME[m]} (output only)", fontsize=10)
        note(f"Confusion {MODEL_NAME[m]} output (rows=true {FAMILIES}): {cm.tolist()}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_confusion.png"))
    plt.close(fig)

    # ---- recall by model VERSION within each family (output only, ensembles)
    rows = []
    ver = df.set_index("uid")["model"]
    for m in ms:
        if (m, "output") not in ens_preds:
            continue
        uid, y, p = ens_preds[(m, "output")]
        d = pd.DataFrame({"version": ver.loc[uid].values, "family": [FAMILIES[i] for i in y],
                          "correct": (y == p)})
        g = d.groupby(["family", "version"])["correct"].agg(["mean", "size"]).reset_index()
        g["model"] = MODEL_NAME[m]
        rows.append(g)
    if rows:
        vt = pd.concat(rows)
        vt = vt.pivot_table(index=["family", "version", "size"], columns="model",
                            values="mean").reset_index()
        vt = vt.rename(columns={"size": "n_test"}).sort_values(["family", "n_test"],
                                                              ascending=[True, False])
        for c in vt.columns[3:]:
            vt[c] = (100 * vt[c]).round(1)
        to_md(vt, os.path.join(TABLE_DIR, "rq1_recall_by_version.md"))
        note("### Recall by model version (output only, seed ensembles)\n" + vt.to_markdown(index=False) + "\n")

    # ---- learning curves (training details)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, m in zip(axes, ["cnn", "lstm"]):
        seed_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
        rs = sorted(runs("rq1_rq2", f"{m}_output_s*.json"), key=lambda r: r["seed"])
        for r, col in zip(rs, seed_colors):
            h = r["history"]
            ax.plot([e["epoch"] for e in h], [e["val_macro_f1"] for e in h], lw=2, color=col,
                    marker="o", ms=4, label=f"seed {r['seed']} (best epoch {r['best_epoch']})")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title(f"{MODEL_NAME[m]} (output only)", fontsize=10, loc="left")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Validation macro-F1")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_learning_curves.png"))
    plt.close(fig)
    return tab


# ==========================================================================
def rq2_confounded(main_tab):
    d = os.path.join(RUNS_DIR, "rq2_confounded")
    if not os.path.exists(os.path.join(d, "design.json")):
        return
    design = load_json(os.path.join(d, "design.json"))
    note("## RQ2 demo: random vs confounded collection (separate pool, see design)")
    note(f"Design: {design}")
    # Upper bound for an input-only classifier that knows the task type
    # perfectly: predict the family with the largest share of that task.
    mix = design["confounded_mix"]
    tasks = list(next(iter(mix.values())).keys())
    ceiling = sum(max(mix[f][t] for f in FAMILIES) for t in tasks) / len(FAMILIES)
    note(f"Theoretical input-only accuracy ceiling under the confounded mix "
         f"(perfect task knowledge): {100 * ceiling:.1f}%")
    rows = []
    for view in ["input", "output"]:
        for m in ["tfidf_lr", "cnn"]:
            vals = {}
            for cond in ["random", "confounded"]:
                pat = f"{cond}_{m}_{view}.json" if m == "tfidf_lr" else f"{cond}_{m}_{view}_s*.json"
                vals[cond] = [r["test"]["macro_f1"] for r in runs("rq2_confounded", pat)]
            if not vals["random"] or not vals["confounded"]:
                continue
            rows.append({"model": MODEL_NAME[m], "view": VIEW_NAME[view],
                         "random collection (macro-F1)": mstd(vals["random"]),
                         "confounded collection (macro-F1)": mstd(vals["confounded"]),
                         "_r": np.mean(vals["random"]), "_c": np.mean(vals["confounded"])})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    to_md(t[[c for c in t.columns if not c.startswith("_")]], os.path.join(TABLE_DIR, "rq2_confounded.md"))
    note(t[[c for c in t.columns if not c.startswith("_")]].to_markdown(index=False) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = [f"{m}\n{v}" for m, v in zip(t.model, t.view)]
    x = np.arange(len(t))
    b1 = ax.bar(x - 0.18, t._r, 0.34, color="#2a78d6", label="Same task mix for every family")
    b2 = ax.bar(x + 0.18, t._c, 0.34, color="#eb6834", label="Task mix differs by family (confounded)")
    bar_labels(ax, b1, t._r)
    bar_labels(ax, b2, t._c)
    ax.axhline(CHANCE, ls="--", lw=1, color=MUTED, label="Chance (25%)")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG_DIR, "fig_rq2_confounded.png"))
    plt.close(fig)


def era_section():
    p = os.path.join(RUNS_DIR, "rq2_era", "era_analysis.json")
    if not os.path.exists(p):
        return
    e = load_json(p)
    note("## RQ2 (iii): does the prompt reveal WHEN it was written? (within GPT + Claude)")
    note(pd.read_csv(os.path.join(TABLE_DIR, "era_by_family.csv")).to_markdown(index=False))
    note(f"Prompt -> era (early vs late) logistic regression on {e['n_train']} train / {e['n_test']} test "
         f"examples: test AUC {e['test_auc']:.3f} (95% CI {e['auc_95ci'][0]:.3f}-{e['auc_95ci'][1]:.3f}); "
         f"label-permutation null mean AUC {e['permutation_null_auc_mean']:.3f}, permutation p = {e['permutation_p']:.3f}")
    note(f"Input-only TF-IDF per-family recall: {e['input_only_tfidf_per_family_recall']}\n")


# ==========================================================================
def rq3():
    if not glob.glob(os.path.join(RUNS_DIR, "rq3", "*.json")):
        return
    design = load_json(os.path.join(RUNS_DIR, "rq3", "design.json"))
    note("## RQ3 cross-task generalization")
    note(f"Design sizes: {design}")
    rows, fam_rows = [], []
    for target in ["writing", "coding"]:
        for m in ["tfidf_lr", "cnn", "lstm"]:
            res = {}
            for cond in ["id", "ood"]:
                pat = f"{target}_{cond}_{m}.json" if m == "tfidf_lr" else f"{target}_{cond}_{m}_s*.json"
                res[cond] = runs("rq3", pat)
            if not res["id"] or not res["ood"]:
                continue
            fi = [r["test"]["macro_f1"] for r in res["id"]]
            fo = [r["test"]["macro_f1"] for r in res["ood"]]
            # paired bootstrap of the change, on seed-ensemble predictions
            u1, y1, p_id = ensemble(res["id"])
            u2, y2, p_ood = ensemble(res["ood"])
            assert (u1 == u2).all()
            diffs = []
            for _ in range(2000):
                s_ = rng.choice(len(y1), len(y1), replace=True)
                diffs.append(macro(y1[s_], p_ood[s_]) - macro(y1[s_], p_id[s_]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({"test task": target, "model": MODEL_NAME[m], "seeds": len(fo),
                         "n_test": len(y1),
                         "trained WITH target task (macro-F1)": mstd(fi),
                         "trained WITHOUT target task (macro-F1)": mstd(fo),
                         "change (points)": f"{100 * (np.mean(fo) - np.mean(fi)):+.1f}",
                         "ensemble change, 95% CI": f"{100 * (macro(y1, p_ood) - macro(y1, p_id)):+.1f} "
                                                    f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                         "_i": np.mean(fi), "_o": np.mean(fo), "_m": m, "_t": target})
            for fam in FAMILIES:
                fam_rows.append({"test task": target, "model": MODEL_NAME[m], "family": fam,
                                 "recall_with": 100 * np.mean([r["test"]["per_class"][fam]["recall"] for r in res["id"]]),
                                 "recall_without": 100 * np.mean([r["test"]["per_class"][fam]["recall"] for r in res["ood"]])})
    t = pd.DataFrame(rows)
    to_md(t[[c for c in t.columns if not c.startswith("_")]], os.path.join(TABLE_DIR, "rq3_main.md"))
    note(t[[c for c in t.columns if not c.startswith("_")]].to_markdown(index=False) + "\n")
    ft = pd.DataFrame(fam_rows)
    ft["change"] = ft.recall_without - ft.recall_with
    ft = ft.round(1)
    to_md(ft, os.path.join(TABLE_DIR, "rq3_per_family.md"))
    note("### RQ3 per-family recall\n" + ft.to_markdown(index=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    for ax, target in zip(axes, ["writing", "coding"]):
        s = t[t._t == target]
        x = np.arange(len(s))
        b1 = ax.bar(x - 0.18, s._i, 0.34, color="#2a78d6", label="Target task seen in training")
        b2 = ax.bar(x + 0.18, s._o, 0.34, color="#eb6834", label="Target task unseen")
        bar_labels(ax, b1, s._i)
        bar_labels(ax, b2, s._o)
        ax.axhline(CHANCE, ls="--", lw=1, color=MUTED, label="Chance (25%)")
        ax.set_xticks(x, s.model)
        ax.set_title(f"Test on {target} prompts", fontsize=10, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Test macro-F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.savefig(os.path.join(FIG_DIR, "fig_rq3.png"))
    plt.close(fig)


# ==========================================================================
ABL_NAME = {"none": "None (original text)", "no_formatting": "Remove formatting",
            "lower_no_punct": "Lowercase + no punctuation",
            "first_50_words": "First 50 words only", "all_three": "All three"}


def rq4():
    note("## RQ4")
    # ---- ablations
    if glob.glob(os.path.join(RUNS_DIR, "rq4_ablation", "*.json")):
        rows = []
        for m in ["tfidf_lr", "cnn", "lstm"]:
            for a in ABL_NAME:
                if a == "none":
                    pat = f"{m}_output.json" if m == "tfidf_lr" else f"{m}_output_s*.json"
                    rs = runs("rq1_rq2", pat)
                else:
                    pat = f"{a}_{m}.json" if m == "tfidf_lr" else f"{a}_{m}_s*.json"
                    rs = runs("rq4_ablation", pat)
                if not rs:
                    continue
                f1 = [r["test"]["macro_f1"] for r in rs]
                rows.append({"ablation": ABL_NAME[a], "model": MODEL_NAME[m], "seeds": len(rs),
                             "macro_f1": mstd(f1), "_f": np.mean(f1), "_a": a, "_m": m})
        t = pd.DataFrame(rows)
        base = t[t._a == "none"].set_index("_m")._f
        t["change (points)"] = [f"{100 * (f - base[m]):+.1f}" for f, m in zip(t._f, t._m)]
        to_md(t[[c for c in t.columns if not c.startswith("_")]], os.path.join(TABLE_DIR, "rq4_ablation.md"))
        note("### Ablations\n" + t[[c for c in t.columns if not c.startswith("_")]].to_markdown(index=False) + "\n")

        fig, ax = plt.subplots(figsize=(7, 3.2))
        abls = [a for a in ABL_NAME if a in set(t._a)]
        ms = [m for m in ["tfidf_lr", "cnn", "lstm"] if m in set(t._m)]
        w = 0.8 / len(ms)
        for i, m in enumerate(ms):
            vals = [t[(t._a == a) & (t._m == m)]._f.iloc[0] if len(t[(t._a == a) & (t._m == m)]) else np.nan
                    for a in abls]
            x = np.arange(len(abls)) + (i - (len(ms) - 1) / 2) * w
            bars = ax.bar(x, vals, w - 0.03, color=MODEL_COLOR[m], label=MODEL_NAME[m])
            bar_labels(ax, bars, vals)
        ax.axhline(CHANCE, ls="--", lw=1, color=MUTED, label="Chance (25%)")
        ax.set_xticks(range(len(abls)), [ABL_NAME[a].replace(" + ", " +\n").replace(" only", "\nonly")
                                         .replace("formatting", "\nformatting") for a in abls], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test macro-F1 (output only)")
        ax.legend(ncol=4, fontsize=8, loc="upper right")
        fig.savefig(os.path.join(FIG_DIR, "fig_rq4_ablation.png"))
        plt.close(fig)

    # ---- feature groups
    p = os.path.join(RUNS_DIR, "rq4_stylometry", "feature_groups.json")
    if os.path.exists(p):
        fg = load_json(p)
        t = pd.DataFrame([{"feature group": g, "n_features": len(v["features"]),
                           "accuracy": round(100 * v["test"]["accuracy"], 1),
                           "macro_f1": round(100 * v["test"]["macro_f1"], 1)} for g, v in fg.items()])
        to_md(t, os.path.join(TABLE_DIR, "rq4_feature_groups.md"))
        note("### Hand-crafted feature groups (gradient boosting)\n" + t.to_markdown(index=False) + "\n")
        fig, ax = plt.subplots(figsize=(6, 2.8))
        t = t.sort_values("macro_f1")
        ax.barh(t["feature group"].str.replace("_", " "), t.macro_f1 / 100, color="#2a78d6", height=0.6)
        for i, v in enumerate(t.macro_f1):
            ax.text(v / 100 + 0.01, i, f"{v:.1f}", va="center", fontsize=8)
        ax.axvline(CHANCE, ls="--", lw=1, color=MUTED)
        ax.set_ylim(-0.6, len(t) + 0.2)
        ax.text(CHANCE + 0.01, len(t) - 0.35, "chance", fontsize=7.5, color=MUTED)
        cnn = runs("rq1_rq2", "cnn_output_s*.json")
        if cnn:
            cf = np.mean([r["test"]["macro_f1"] for r in cnn])
            ax.axvline(cf, lw=1.2, color="#eb6834")
            ax.text(cf + 0.01, len(t) - 0.35, f"CNN on full text ({100 * cf:.1f})", fontsize=7.5, color=INK)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Test macro-F1")
        ax.grid(axis="y", visible=False)
        fig.savefig(os.path.join(FIG_DIR, "fig_rq4_feature_groups.png"))
        plt.close(fig)

    # ---- style profile figure + descriptive table
    fp = os.path.join(TABLE_DIR, "stylometry_features_all.csv.gz")
    if os.path.exists(fp):
        F = pd.read_csv(fp)
        desc = pd.read_csv(os.path.join(TABLE_DIR, "stylometry_by_family.csv"))
        keep = ["feature"] + [f"{f}_median" for f in FAMILIES] + [f"{f}_mean" for f in FAMILIES] + ["epsilon_sq", "p_value"]
        d2 = desc[keep].copy()
        for c in keep[1:-1]:
            d2[c] = d2[c].round(3)
        d2["p_value"] = d2["p_value"].map(lambda v: f"{v:.1e}")
        to_md(d2, os.path.join(TABLE_DIR, "rq4_stylometry_by_family.md"))
        note("### Stylometric features by family (sorted by Kruskal-Wallis effect size)\n" + d2.to_markdown(index=False) + "\n")

        panels = [("n_words", "Median length (words)", "median"),
                  ("mtld", "Lexical diversity (MTLD, median)", "median"),
                  ("headers", "% responses with markdown headers", "share"),
                  ("bullet_lines", "% responses with bullet lists", "share"),
                  ("numbered_lines", "% responses with numbered lists", "share"),
                  ("bold_spans", "% responses with bold text", "share"),
                  ("opens_sure_certainly", "% opening 'Sure/Certainly/…'", "share"),
                  ("closes_offer_help", "% closing with an offer of help", "share")]
        fig, axes = plt.subplots(2, 4, figsize=(10, 4.4))
        for ax, (col, title, kind) in zip(axes.flat, panels):
            vals = [(F.loc[F.family == f, col].median() if kind == "median"
                     else 100 * (F.loc[F.family == f, col] > 0).mean()) for f in FAMILIES]
            bars = ax.bar(FAMILIES, vals, color=[FAM_COLOR[f] for f in FAMILIES], width=0.65)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
            ax.set_title(title, fontsize=8.5, loc="left")
            ax.tick_params(axis="x", labelsize=7.5)
            ax.tick_params(axis="y", labelsize=7)
            ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_rq4_style_profiles.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 2.8))
        data = [np.log10(F.loc[F.family == f, "n_words"].clip(lower=1)) for f in FAMILIES]
        bp = ax.boxplot(data, orientation="horizontal", tick_labels=FAMILIES, patch_artist=True, widths=0.55,
                        showfliers=False, medianprops=dict(color=INK, lw=1.5))
        for patch, f in zip(bp["boxes"], FAMILIES):
            patch.set_facecolor(FAM_COLOR[f])
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        ticks = [1, 10, 30, 100, 300, 1000]
        ax.set_xticks(np.log10(ticks), [str(v) for v in ticks])
        ax.set_xlabel("Response length in words (log scale; boxes = IQR, whiskers = 1.5 IQR)")
        ax.grid(axis="y", visible=False)
        fig.savefig(os.path.join(FIG_DIR, "fig_rq4_length.png"))
        plt.close(fig)

    for name in ("rq4_lr_top_ngrams", "rq4_cnn_saliency_tokens"):
        p = os.path.join(TABLE_DIR, name + ".csv")
        if os.path.exists(p):
            t = pd.read_csv(p)
            col = "ngram" if "ngram" in t else "token"
            wide = pd.DataFrame({f: t[t.family == f].sort_values("rank")[col].head(15).tolist()
                                 for f in FAMILIES})
            wide.index = range(1, len(wide) + 1)
            with open(os.path.join(TABLE_DIR, name + "_wide.md"), "w") as fh:
                fh.write(wide.to_markdown())
            note(f"### {name} (top 15 per family)\n" + wide.to_markdown() + "\n")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = data_section()
    tab = rq1_rq2(df)
    rq2_confounded(tab)
    era_section()
    rq3()
    rq4()
    with open(os.path.join(os.path.dirname(FIG_DIR), "SUMMARY.md"), "w") as f:
        f.write("# Results summary (auto-generated by src/analyze.py)\n\n" + "\n".join(SUMMARY))
    print("wrote results/SUMMARY.md, tables and figures")


if __name__ == "__main__":
    main()
