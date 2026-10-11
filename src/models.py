"""The two required neural models (CNN and RNN/BiLSTM) plus the wrapper that
lets either one read the response, the prompt, or both (RQ2).

Shapes: token ids are LongTensor [batch, seq_len] padded with 0; `lengths`
holds the true length of each sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim (2014)-style CNN text encoder.

    embedding -> parallel 1-D convolutions with window sizes 3/4/5 (n-gram
    detectors) -> ReLU -> max-pool over time (masked so padding is ignored)
    -> concatenated feature vector of size n_filters * len(kernel_sizes).
    """

    def __init__(self, vocab_size, emb_dim=128, n_filters=128, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.out_dim = n_filters * len(kernel_sizes)

    def forward(self, ids, lengths, return_emb=False):
        e = self.emb(ids)                      # [B, T, E]
        x = e.transpose(1, 2)                  # [B, E, T]
        feats = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            h = F.relu(conv(x))                # [B, F, T-k+1]
            # position j is a real n-gram only if j <= length - k
            n_valid = (lengths - k + 1).clamp(min=1)
            pos = torch.arange(h.size(2), device=h.device)
            mask = pos[None, :] < n_valid[:, None]
            h = h.masked_fill(~mask[:, None, :], float("-inf"))
            feats.append(h.max(dim=2).values)  # [B, F]
        out = torch.cat(feats, dim=1)
        return (out, e) if return_emb else out


def _reverse_valid(x, lengths):
    """Reverse each sequence's first `length` steps in place, leaving the
    padding at the end: [a b c PAD] -> [c b a PAD]. Applying it twice is the
    identity."""
    B, T = x.shape[:2]
    pos = torch.arange(T, device=x.device)[None, :].expand(B, T)
    L = lengths[:, None]
    idx = torch.where(pos < L, L - 1 - pos, pos)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder with additive attention pooling.

    embedding -> BiLSTM -> attention weights over time steps (masked
    softmax) -> weighted sum of hidden states.

    The two directions are two separate single-layer LSTMs. The backward
    LSTM reads each sequence reversed *within its true length*, so neither
    direction ever reads padding before real tokens. This is mathematically
    the same as a packed bidirectional nn.LSTM, but avoids PyTorch's very
    slow CPU backward pass for packed sequences of unequal length (20+ s per
    batch on our machine vs. ~0.5 s).
    """

    def __init__(self, vocab_size, emb_dim=128, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        assert layers == 1, "only single-layer BiLSTM is implemented"
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm_fwd = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.lstm_bwd = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.att = nn.Linear(2 * hidden, 64)
        self.att_v = nn.Linear(64, 1, bias=False)
        self.out_dim = 2 * hidden

    def forward(self, ids, lengths, return_emb=False):
        e = self.emb(ids)
        x = self.emb_drop(e)
        h_f, _ = self.lstm_fwd(x)                          # left -> right
        h_b, _ = self.lstm_bwd(_reverse_valid(x, lengths))  # right -> left
        h = torch.cat([h_f, _reverse_valid(h_b, lengths)], dim=-1)  # [B, T, 2H]
        scores = self.att_v(torch.tanh(self.att(h))).squeeze(-1)       # [B, T]
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1)
        pooled = (w.unsqueeze(-1) * h).sum(dim=1)                       # [B, 2H]
        return (pooled, e) if return_emb else pooled


ENCODERS = {"cnn": TextCNN, "lstm": BiLSTMEncoder}


class FamilyClassifier(nn.Module):
    """One encoder per text field in `fields` ("response", "prompt" or both).

    For the Input+Output setting (RQ2) the prompt and the response get their
    own encoders (same architecture, separate weights) and the two pooled
    vectors are concatenated before the classification layer.
    """

    def __init__(self, model_type, fields, vocab_size, n_classes=4, dropout=0.5, **enc_kwargs):
        super().__init__()
        self.fields = list(fields)
        self.encoders = nn.ModuleDict(
            {f: ENCODERS[model_type](vocab_size, **enc_kwargs) for f in self.fields})
        dim = sum(enc.out_dim for enc in self.encoders.values())
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, batch):
        feats = [self.encoders[f](batch[f], batch[f + "_len"]) for f in self.fields]
        return self.fc(self.drop(torch.cat(feats, dim=1)))
