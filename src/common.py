"""Shared constants and small helpers used across the project."""
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

RAW_PATH = os.path.join(DATA_DIR, "raw_4families.csv.gz")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.csv.gz")

FAMILIES = ["GPT", "Claude", "Gemini", "Llama"]  # fixed label order everywhere
LABEL2ID = {f: i for i, f in enumerate(FAMILIES)}

SEEDS = [13, 42, 2026]  # the three seeds used for every neural experiment


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, RUNS_DIR, FIG_DIR, TABLE_DIR):
        os.makedirs(d, exist_ok=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
