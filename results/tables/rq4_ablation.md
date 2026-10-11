| ablation                   | model       |   seeds | macro_f1   |   change (points) |
|:---------------------------|:------------|--------:|:-----------|------------------:|
| None (original text)       | TF-IDF + LR |       1 | 74.6       |               0   |
| Remove formatting          | TF-IDF + LR |       1 | 71.2       |              -3.4 |
| Lowercase + no punctuation | TF-IDF + LR |       1 | 66.5       |              -8.1 |
| First 50 words only        | TF-IDF + LR |       1 | 68.4       |              -6.2 |
| All three                  | TF-IDF + LR |       1 | 61.5       |             -13.1 |
| None (original text)       | CNN         |       3 | 74.4 ± 2.9 |               0   |
| Remove formatting          | CNN         |       3 | 64.4 ± 1.4 |             -10   |
| Lowercase + no punctuation | CNN         |       3 | 62.1 ± 3.2 |             -12.3 |
| First 50 words only        | CNN         |       3 | 67.8 ± 0.7 |              -6.6 |
| All three                  | CNN         |       3 | 59.1 ± 2.1 |             -15.3 |
| None (original text)       | BiLSTM      |       3 | 67.8 ± 2.2 |               0   |
| Remove formatting          | BiLSTM      |       3 | 62.3 ± 0.2 |              -5.4 |
| Lowercase + no punctuation | BiLSTM      |       3 | 63.2 ± 0.9 |              -4.6 |
| First 50 words only        | BiLSTM      |       3 | 67.0 ± 2.3 |              -0.8 |
| All three                  | BiLSTM      |       3 | 59.4 ± 0.3 |              -8.4 |