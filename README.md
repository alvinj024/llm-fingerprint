# Who Wrote It? Identifying LLMs from Their Responses
# Alvin James, Aryan Patel, Anurag Palakurti, Somay Bansal

CMPSC 448 midterm project. We train a **CNN** and a **BiLSTM** to identify which LLM
family (**GPT, Claude, Gemini, Llama**) wrote a response, and study:

| RQ | Question | Where |
|----|----------|-------|
| RQ1 | Can we identify the LLM from its output? | `src/run_experiments.py --stage rq1_rq2` |
| RQ2 | Does the user's prompt help? (input only / output only / input + output) | same stage, plus `src/era_analysis.py` and `--stage rq2_confounded` |
| RQ3 | Do fingerprints generalize to an unseen task type? | `--stage rq3` |
| RQ4 | What characteristics distinguish the LLMs? | `--stage rq4`, `src/stylometry.py`, `src/interpret.py` |

Everything runs on a CPU (no GPU needed).

## Data

* **Source:** [LMSYS Chatbot Arena human-preference data](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k)
  (`train.csv`, ~55k rows). Each row is one real user prompt answered by **two models
  that the Arena assigned at random**. We split every row into two
  `(LLM_name, LLM_input, LLM_output)` examples.
* **`scripts/download_and_filter.py`** downloads the file, keeps single-turn conversations
  with a non-empty prompt and response, and maps model versions to families with explicit
  rules (third-party fine-tunes such as Vicuna, and CodeLlama, are excluded).
  Output: `data/raw_4families.csv.gz` (committed) and `data/model_counts.csv`.
* **`src/data.py`**: English filter → exact-duplicate removal → task-type label
  → **1,000 random examples per family** → 70/15/15 train/val/test split **grouped by
  prompt** (an identical prompt never appears in two splits; the code asserts this).

## Reproduce everything

```bash
pip install -r requirements.txt
# (only if data/raw_4families.csv.gz is missing)
python scripts/download_and_filter.py
./run_all.sh            # ~4-5 h on a 2-core CPU; every stage is resumable
```

`run_all.sh` runs, in order:

| Step | Command | Output |
|------|---------|--------|
| Data prep | `python -m src.data` | `data/processed.csv.gz`, `results/tables/data_*.csv` |
| Task-label audit | `python -m src.check_task_labels --score` | `results/tables/task_label_precision.csv` |
| RQ1 + RQ2 | `python -m src.run_experiments --stage rq1_rq2` | `results/runs/rq1_rq2/*.json` |
| RQ2 (iii) | `python -m src.era_analysis` | `results/runs/rq2_era/era_analysis.json` |
| RQ2 demo | `python -m src.run_experiments --stage rq2_confounded` | `results/runs/rq2_confounded/` |
| RQ3 | `python -m src.run_experiments --stage rq3` | `results/runs/rq3/` |
| RQ4 ablations | `python -m src.run_experiments --stage rq4` | `results/runs/rq4_ablation/` |
| RQ4 features | `python -m src.stylometry` | `results/tables/stylometry_*.csv` |
| RQ4 interpretation | `python -m src.interpret` | `results/tables/rq4_*` |
| Tables + figures | `python -m src.analyze` | `results/tables/*.md`, `results/figures/*.png`, `results/SUMMARY.md` |

A single model can also be trained directly, e.g.
`python -m src.train --model lstm --view output --seed 13` (prints the per-epoch log).

## Models (`src/models.py`, `src/train.py`)

* **CNN**: Kim-style TextCNN. 128-d embeddings, filter widths 3/4/5 × 128 filters, ReLU,
  masked max-pooling over time, dropout 0.5, linear output layer.
* **BiLSTM**: 128-d embeddings (dropout 0.3), single-layer bidirectional LSTM (128 units
  per direction), additive attention pooling, dropout 0.5 before the output layer. The two
  directions are two `nn.LSTM`s (the backward one reads each sequence reversed within its true
  length): identical to a packed bidirectional LSTM but ~17x faster on CPU
  (see `tests/test_bilstm_equivalence.py`).
* **Input + Output**: separate encoders (same architecture) for the prompt and the
  response; the two pooled vectors are concatenated before the output layer.
* **Tokenizer** keeps case, punctuation, line breaks and markdown symbols. The vocabulary
  is built from the training split only (min frequency 2, max 30k tokens). Responses are
  truncated to 512 tokens, prompts to 256.
* **Training**: AdamW (lr 1e-3, weight decay 1e-4), batch 64, gradient clipping 1.0,
  up to 30 epochs, early stopping on validation macro-F1 (patience 3), 3 seeds
  (13, 42, 2026).
* **Baselines**: majority class; TF-IDF (uni+bigrams, same tokenizer) + logistic
  regression with C tuned on the validation set.

## Tests

```bash
python tests/verify_results.py            # recompute every metric from saved predictions, leakage checks
python tests/test_bilstm_equivalence.py   # our BiLSTM == PyTorch packed BiLSTM
python tests/make_synthetic_data.py /tmp/synth.csv.gz 3000   # planted-signal data
```

`tests/make_synthetic_data.py` builds fake data in which each family has a known quirk.
We ran the full pipeline on it before using real data, to check that (a) the models
learn the planted signals, (b) input-only accuracy is at chance, and (c) the RQ4
analyses recover exactly the planted quirks.

## Repository layout

```
scripts/download_and_filter.py   step 0: raw Arena data -> 4-family examples
src/  common.py textproc.py      constants; tokenizer, task rules, ablation transforms
      data.py                    cleaning, balancing, grouped split
      models.py train.py         CNN, BiLSTM, training loop, metrics
      baselines.py               majority, TF-IDF + logistic regression
      run_experiments.py         all experiments (resumable)
      era_analysis.py            RQ2 (iii)
      stylometry.py interpret.py RQ4
      analyze.py                 tables, figures, SUMMARY.md
      check_task_labels.py       manual audit of task labels
results/                         run JSONs, tables, figures (generated)
tests/                           equivalence test, synthetic data generator
```

## Logs

`logs/run_all.log` is the console output of the full run. The TF-IDF baselines were
re-run afterwards with a wider regularization grid (C = 0.01 to 1,000), so their numbers in
`logs/tfidf_rerun.log` replace the TF-IDF lines in `run_all.log`. The JSON files in
`results/runs/` always hold the final numbers.
