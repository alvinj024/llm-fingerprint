"""Checks that our two-LSTM BiLSTM gives the same hidden states as PyTorch's
packed bidirectional nn.LSTM (with copied weights) on padded batches of
unequal lengths.   Run:  python tests/test_bilstm_equivalence.py"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import BiLSTMEncoder, _reverse_valid  # noqa: E402


def main():
    torch.manual_seed(0)
    enc = BiLSTMEncoder(100, emb_dim=8, hidden=6).eval()
    ref = nn.LSTM(8, 6, batch_first=True, bidirectional=True)
    with torch.no_grad():
        for n in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            getattr(ref, n + "_l0").copy_(getattr(enc.lstm_fwd, n + "_l0"))
            getattr(ref, n + "_l0_reverse").copy_(getattr(enc.lstm_bwd, n + "_l0"))
    ids = torch.randint(1, 100, (5, 9))
    L = torch.tensor([9, 4, 1, 7, 2])
    for i, l in enumerate(L):
        ids[i, l:] = 0
    with torch.no_grad():
        x = enc.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(x, L, batch_first=True, enforce_sorted=False)
        o, _ = ref(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=9)
        hf, _ = enc.lstm_fwd(x)
        hb, _ = enc.lstm_bwd(_reverse_valid(x, L))
        h = torch.cat([hf, _reverse_valid(hb, L)], -1)
    mask = (torch.arange(9)[None, :] < L[:, None]).unsqueeze(-1)
    diff = float(((h - o).abs() * mask).max())
    assert diff < 1e-5, diff
    print(f"OK: max abs difference on real tokens = {diff:.2e}")


if __name__ == "__main__":
    main()
