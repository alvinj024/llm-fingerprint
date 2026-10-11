"""
Step 0 of the pipeline: download the public LMSYS Chatbot Arena data and
reduce it to the (LLM_name, LLM_input, LLM_output) triples we use.

Source dataset
--------------
lmsys/lmsys-arena-human-preference-55k  (Hugging Face, file: train.csv)
Each row = one real user prompt sent to Chatbot Arena, answered by two
models (model_a, model_b) that the Arena assigned at random. Columns
`prompt`, `response_a`, `response_b` are JSON-encoded lists (one entry per
conversation turn).

What this script does
---------------------
1. Downloads train.csv (or reads a local copy passed with --csv).
2. Splits every row into two examples: (model_a, prompt, response_a) and
   (model_b, prompt, response_b).
3. Keeps single-turn conversations only, with a non-empty prompt and response.
4. Maps model versions to the four families used in the project
   (GPT, Claude, Gemini, Llama) using explicit rules (see FAMILY_RULES).
   Every other model (Mistral, Vicuna, Qwen, ...) is dropped.
5. Writes:
     data/raw_4families.csv.gz  - the kept examples
     data/model_counts.csv      - every model name in the source and what
                                  happened to it (for the report's data section)

Usage
-----
    pip install pandas huggingface_hub
    python scripts/download_and_filter.py                 # downloads automatically
    python scripts/download_and_filter.py --csv train.csv # if you downloaded it by hand
"""
import argparse
import json
import os
import re

import pandas as pd

REPO_ID = "lmsys/lmsys-arena-human-preference-55k"
FILENAME = "train.csv"

# Explicit, documented family rules. Order matters only for readability;
# the patterns are mutually exclusive.
FAMILY_RULES = [
    # OpenAI GPT-3.5 / GPT-4 chat models (excludes e.g. "gpt4all").
    ("GPT", re.compile(r"^gpt-(3\.5|4)")),
    # Anthropic Claude models (claude-1, claude-2.x, claude-instant-1, claude-3-*).
    ("Claude", re.compile(r"^claude-")),
    # Google Gemini models (gemini-pro, gemini-pro-dev-api, ...).
    ("Gemini", re.compile(r"^gemini-")),
    # Meta's official general-purpose chat/instruct Llama releases only.
    # Excludes third-party fine-tunes (e.g. llama2-70b-steerlm-chat), the
    # original non-chat LLaMA (llama-13b) and the code-specialised CodeLlama,
    # so that "Llama" means Meta's own chat model, comparable to the others.
    ("Llama", re.compile(r"^llama-(2-\d+b-chat|3-\d+b-instruct)$")),
]


def family_of(model_name: str):
    for fam, pat in FAMILY_RULES:
        if pat.search(model_name):
            return fam
    return None


def parse_turns(cell):
    """The Arena columns are JSON lists of strings; return the list or None."""
    if not isinstance(cell, str):
        return None
    try:
        turns = json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return None
    return turns if isinstance(turns, list) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to a local train.csv (skip download)")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    if args.csv:
        path = args.csv
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    print(f"Reading {path}")
    src = pd.read_csv(path)
    print(f"Source rows (prompt pairs): {len(src):,}")

    rows, reasons = [], []
    for _, r in src.iterrows():
        prompts = parse_turns(r["prompt"])
        for side in ("a", "b"):
            model = str(r[f"model_{side}"])
            resps = parse_turns(r[f"response_{side}"])
            fam = family_of(model)
            if prompts is None or resps is None:
                reason = "unparseable"
            elif len(prompts) != 1 or len(resps) != 1:
                reason = "multi_turn"
            elif not isinstance(prompts[0], str) or not prompts[0].strip():
                reason = "empty_prompt"
            elif not isinstance(resps[0], str) or not resps[0].strip():
                reason = "empty_response"
            elif fam is None:
                reason = "other_family"
            else:
                reason = "kept"
                rows.append({
                    "pair_id": int(r["id"]),
                    "side": side,
                    "model": model,
                    "family": fam,
                    "prompt": prompts[0],
                    "response": resps[0],
                })
            reasons.append({"model": model, "family": fam or "", "reason": reason})

    os.makedirs(args.out_dir, exist_ok=True)
    kept = pd.DataFrame(rows)
    kept.to_csv(os.path.join(args.out_dir, "raw_4families.csv.gz"), index=False, compression="gzip")

    rs = pd.DataFrame(reasons)
    counts = rs.pivot_table(index=["model", "family"], columns="reason", aggfunc="size", fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    counts.to_csv(os.path.join(args.out_dir, "model_counts.csv"))

    print(f"\nKept {len(kept):,} examples. By family:")
    print(kept["family"].value_counts().to_string())
    print("\nDrop reasons over all responses:")
    print(rs["reason"].value_counts().to_string())
    print(f"\nWrote {args.out_dir}/raw_4families.csv.gz and {args.out_dir}/model_counts.csv")


if __name__ == "__main__":
    main()
