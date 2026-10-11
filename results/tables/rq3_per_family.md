| test task   | model       | family   |   recall_with |   recall_without |   change |
|:------------|:------------|:---------|--------------:|-----------------:|---------:|
| writing     | TF-IDF + LR | GPT      |          57.1 |             39.3 |    -17.9 |
| writing     | TF-IDF + LR | Claude   |         100   |             97.3 |     -2.7 |
| writing     | TF-IDF + LR | Gemini   |          58.3 |             61.1 |      2.8 |
| writing     | TF-IDF + LR | Llama    |          80   |             80   |      0   |
| writing     | CNN         | GPT      |          65.5 |             64.3 |     -1.2 |
| writing     | CNN         | Claude   |          98.2 |             93.7 |     -4.5 |
| writing     | CNN         | Gemini   |          63   |             63   |      0   |
| writing     | CNN         | Llama    |          67.6 |             67.6 |      0   |
| writing     | BiLSTM      | GPT      |          48.8 |             39.3 |     -9.5 |
| writing     | BiLSTM      | Claude   |          86.5 |             88.3 |      1.8 |
| writing     | BiLSTM      | Gemini   |          67.6 |             58.3 |     -9.3 |
| writing     | BiLSTM      | Llama    |          71.4 |             67.6 |     -3.8 |
| coding      | TF-IDF + LR | GPT      |          65.8 |             68.4 |      2.6 |
| coding      | TF-IDF + LR | Claude   |          83.3 |             80.6 |     -2.8 |
| coding      | TF-IDF + LR | Gemini   |          46.2 |             64.1 |     17.9 |
| coding      | TF-IDF + LR | Llama    |          69.2 |             69.2 |      0   |
| coding      | CNN         | GPT      |          63.2 |             58.8 |     -4.4 |
| coding      | CNN         | Claude   |          87   |             84.3 |     -2.8 |
| coding      | CNN         | Gemini   |          80.3 |             53   |    -27.4 |
| coding      | CNN         | Llama    |          65   |             72.6 |      7.7 |
| coding      | BiLSTM      | GPT      |          51.8 |             59.6 |      7.9 |
| coding      | BiLSTM      | Claude   |          69.4 |             79.6 |     10.2 |
| coding      | BiLSTM      | Gemini   |          57.3 |             45.3 |    -12   |
| coding      | BiLSTM      | Llama    |          57.3 |             46.2 |    -11.1 |