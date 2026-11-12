# Small language models under a 30-minute compute budget

GPT-style language models are trained from random initialisation (no pretrained weights or
tokenizers) on the TinyStories "all data" archive from Hugging Face. After cleaning, the stories are
split by whole story into 10,000 validation, 10,000 test and 4.95M training stories, and encoded with
either a 4,096-token byte-level BPE trained on the training split or a byte-level tokenizer (UTF-8
bytes). Every run trains for 30 minutes on one RTX 3060 Ti and is scored by bits per byte on the
test split. The trials compare model depth (4, 8 and 12 layers, plus a seed repeat), width versus
FFN size at equal parameters, BPE versus byte-level tokenization at two depths, a
warmup-stable-decay schedule with GPT-4-only stories in the final 20% of training, and BPE versus
byte-level on a 5% subset of the training data. Each configuration's peak learning rate is its
learning-rate range test's rise point divided by 3. Each run writes its log and plots to
`runs/<stage>/<run_name>/`.

## Commands

Native Windows, PowerShell, an NVIDIA GPU with at least 8 GB.

```powershell
# Setup
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q

# Data and tokenizers
.\.venv\Scripts\python.exe -m src.data_prep
.\.venv\Scripts\python.exe -m src.tokenizer --tokenizer bpe --vocab-size 4096
.\.venv\Scripts\python.exe -m src.tokenizer --tokenizer char

# Learning-rate range test for a configuration (example: 8 layers)
.\.venv\Scripts\python.exe -m src.lr_find --set stage=lr run_name=L8 n_layers=8

# E1: depth (4 layers is the baseline)
.\.venv\Scripts\python.exe -m src.train --final --set stage=E1 run_name=L4
.\.venv\Scripts\python.exe -m src.train --final --set stage=E1 run_name=L8 n_layers=8 lr=1.6e-3
.\.venv\Scripts\python.exe -m src.train --final --set stage=E1 run_name=L12 n_layers=12 lr=1.3e-3
.\.venv\Scripts\python.exe -m src.train --final --set stage=E1 run_name=L8_seed1 n_layers=8 lr=1.6e-3 seed=1

# E2: width vs FFN size at equal parameters (8 layers)
.\.venv\Scripts\python.exe -m src.train --final --set stage=E2 run_name=L8_w448_f416 n_layers=8 d_model=448 ffn_size=416 lr=1.4e-3
.\.venv\Scripts\python.exe -m src.train --final --set stage=E2 run_name=L8_w320_f1216 n_layers=8 d_model=320 ffn_size=1216 lr=1.7e-3

# E3 / I1: byte-level tokenizer at 8 and 4 layers
.\.venv\Scripts\python.exe -m src.train --final --set stage=E3 run_name=L8_char_ctx1024 n_layers=8 tokenizer=char context_length=1024 lr=1.1e-3
.\.venv\Scripts\python.exe -m src.train --final --set stage=I1 run_name=L4_char_ctx1024 n_layers=4 tokenizer=char context_length=1024 lr=1.9e-3

# E5: warmup-stable-decay schedule, GPT-4-only stories during the decay
.\.venv\Scripts\python.exe -m src.train --final --set stage=E5 run_name=wsd_gpt4 n_layers=8 schedule=wsd decay_data=gpt4 lr=1.6e-3

# D5: 5% training subset (same validation and test stories)
.\.venv\Scripts\python.exe -m src.data_subset --fraction 0.05 --out data_5pct
.\.venv\Scripts\python.exe -m src.tokenizer --tokenizer bpe --vocab-size 4096 --data-dir data_5pct
.\.venv\Scripts\python.exe -m src.tokenizer --tokenizer char --data-dir data_5pct
.\.venv\Scripts\python.exe -m src.train --final --set stage=D5 run_name=L4_bpe data_dir=data_5pct lr=2.0e-3
.\.venv\Scripts\python.exe -m src.train --final --set stage=D5 run_name=L4_char_ctx1024 data_dir=data_5pct n_layers=4 tokenizer=char context_length=1024 lr=1.8e-3
```

`--final` also evaluates the test split once at the end of the run. The scripts in `scripts/`
(extra evaluations, plots, FLOPs table) give their usage at the top of each file.
