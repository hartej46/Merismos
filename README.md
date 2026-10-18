# Merismos

Merismos is an experimental, decoder-only language-model project built with
PyTorch. It combines programming examples, science question-answering data,
and Tiny Shakespeare into one text corpus, trains a small byte-pair encoding
(BPE) tokenizer, and uses the resulting tokens to train and compare:

1. A conventional GPT-style Transformer baseline.
2. A sparse Mixture-of-Experts (MoE) GPT-style Transformer.

The repository also contains a small SQLite-backed memory layer for storing
chat conversations, messages, and user memories.

> **Project status:** This is a research and learning project. The main
> workflow is currently notebook-driven, and the repository does not yet
> provide a polished command-line inference or chat application connected to
> the trained model.

## How the project works

The complete pipeline is:

```text
Download source datasets
        |
        v
Combine selected text fields into dataset/combined_corpus.txt
        |
        v
Train MinBPE with a 1,500-token vocabulary
        |
        v
Encode the corpus into trained_tokenizer/train.bin
        |
        v
Split tokens into 80% training and 20% validation data
        |
        v
Create 256-token input/target windows
        |
        v
Train the MoE GPT with next-token prediction
```

For every training example, the model receives a sequence `x` and learns to
predict the same sequence shifted one token to the right, `y`. The objective
is causal language modeling: each position can attend only to itself and
tokens before it.

The notebook in [`main.ipynb`](./main.ipynb) is the reference end-to-end
training workflow.

## Main features

- Custom byte-level BPE tokenizer implemented in
  [`tokenizer/BPT_tokenizer.py`](./tokenizer/BPT_tokenizer.py).
- A standard GPT-style baseline in
  [`model_code/baseline_model.py`](./model_code/baseline_model.py).
- A sparse top-1 Mixture-of-Experts model in
  [`model_code/moe_gpt.py`](./model_code/moe_gpt.py).
- A learned router and four feed-forward experts in
  [`model_code/router.py`](./model_code/router.py),
  [`model_code/experts.py`](./model_code/experts.py), and
  [`model_code/SparseMoEFeedForward.py`](./model_code/SparseMoEFeedForward.py).
- Reusable training and validation loop in
  [`loops_for_model/training_loops.py`](./loops_for_model/training_loops.py).
- Automatic device selection for Apple Silicon, CUDA, or CPU in
  [`deviceConfig.py`](./deviceConfig.py).
- SQLite persistence for conversations, messages, and explicit memories in
  [`memory/`](./memory/).

## Dataset

The project uses three sources. The checked-in CSV and text files are the
downloaded/generated copies used by the current repository.

### 1. Python code dataset

- **Hugging Face dataset:** `flytech/python-codes-25k`
- **Local file:** [`dataset/code_csv/train.csv`](./dataset/code_csv/train.csv)
- **Rows in the current file:** 49,626
- **Columns:** `output`, `instruction`, `input`, and `text`
- **Field used in the corpus:** `text`

The `text` field combines a programming instruction, an input/context
description, and a Python output example. It supplies code-oriented language
and natural-language programming instructions.

### 2. Science question dataset

- **Hugging Face dataset loaded by the code:** `allenai/sciq`
- **Local file:** [`dataset/arxiv_csv/arxiv.csv`](./dataset/arxiv_csv/arxiv.csv)
- **Rows in the current file:** 11,679
- **Columns:** `question`, `distractor3`, `distractor1`, `distractor2`,
  `correct_answer`, and `support`
- **Field used in the corpus:** `question`

The directory is named `arxiv_csv`, but the downloader currently loads the
SciQ dataset, not an arXiv dataset. The corpus builder currently extracts
only the `question` column; the answer choices and supporting explanations
are not included.

### 3. Tiny Shakespeare

- **Local file:** [`dataset/input.txt`](./dataset/input.txt)
- **Source URL used by the downloader:**  
  `https://raw.githubusercontent.com/yay1503/GPT-2/refs/heads/main/input.txt`
- **Current file size:** about 1.1 MB

This provides literary English and dialogue-style text.

### Corpus construction

[`dataset_concat_functions.py`](./dataset_concat_functions.py) loads:

1. All of `dataset/input.txt`.
2. The `question` column from CSV files in `dataset/arxiv_csv/`.
3. The `text` column from CSV files in `dataset/code_csv/`.

Documents are joined with the boundary marker:

```text
\n<|endoftext|>\n
```

The current generated corpus is
[`dataset/combined_corpus.txt`](./dataset/combined_corpus.txt), approximately
27.2 MB. Missing directories, missing columns, unreadable CSV files, and
empty values are handled by the corpus-building script with warnings.

## Tokenizer

[`MinBPE`](./tokenizer/BPT_tokenizer.py) is a custom byte-level BPE tokenizer:

1. Text is split with a Unicode-aware regular expression.
2. Each word-like segment is converted to UTF-8 bytes.
3. The initial vocabulary contains all 256 possible byte values.
4. The most frequent adjacent byte/token pair is repeatedly merged.
5. The first merge receives token ID 256, and so on.
6. Encoding applies learned merges in their training order.
7. Decoding joins token byte sequences and decodes them as UTF-8.

The tokenizer is trained with a vocabulary size of 1,500, which means up to
1,244 merges beyond the 256-byte base vocabulary. The training script writes:

- [`trained_tokenizer/moe_tokenizer.json`](./trained_tokenizer/moe_tokenizer.json):
  the learned pair-to-token merge table.
- [`trained_tokenizer/train.bin`](./trained_tokenizer/train.bin):
  the full corpus encoded as a one-dimensional `torch.long` tensor.

To retrain it after rebuilding the corpus:

```bash
python tokenizer/train_tokenizer.py
```

Run this command from the repository root. The script expects
`dataset/combined_corpus.txt` to exist.

## Model architecture

### Shared Transformer components

Both models use causal self-attention with:

- Embedding size: `384`
- Context length: `256` tokens
- Attention heads: `4`
- Dropout: `0.2`
- Token-by-token autoregressive generation support in the baseline model

The attention mask is lower triangular, so a token cannot look ahead at
future target tokens.

### Baseline GPT

[`model_code/baseline_model.py`](./model_code/baseline_model.py) contains a
six-block Transformer. Each block has:

1. Layer normalization.
2. Four-head causal self-attention.
3. A residual connection.
4. Layer normalization.
5. A two-layer ReLU feed-forward network.
6. A second residual connection.

The baseline file currently uses a large hard-coded vocabulary size of
`1,115,394`, matching the byte size of the checked-in Shakespeare input rather
than the 1,500-token tokenizer used by the MoE workflow. Treat it as an
experimental comparison model; it is not directly compatible with the
current `train.bin` token IDs without aligning the vocabulary configuration.

### Sparse Mixture-of-Experts GPT

[`model_code/moe_gpt.py`](./model_code/moe_gpt.py) contains the model used by
the notebook. It has:

- Four Transformer blocks.
- Four feed-forward experts in every block.
- A learned linear router for each block.
- Top-1 routing: each token is sent to the expert with the largest router
  probability.
- Capacity factor `1.25`: an expert processes at most
  `1.25 * (tokens / number_of_experts)` routed tokens per batch.
- Router gate weighting when expert outputs are returned.
- An auxiliary load-balancing loss, weighted by `alpha = 0.01`.
- Final layer normalization and a linear language-model head.

The total loss is:

```text
cross_entropy_loss + 0.01 * load_balancing_loss
```

The load-balancing term encourages the router to distribute tokens across
experts instead of sending nearly every token to one expert.

## Training workflow

### 1. Create or refresh the downloaded datasets

```bash
python dataDownload.py
```

This downloads:

- `flytech/python-codes-25k` into `dataset/code_csv/train.csv`
- Tiny Shakespeare into `dataset/input.txt`
- `allenai/sciq` into `dataset/arxiv_csv/arxiv.csv`

The script uses a local Hugging Face cache under `dataset/.cache_code`.

### 2. Build the combined corpus

```bash
python dataset_concat_functions.py
```

This creates or replaces `dataset/combined_corpus.txt`.

### 3. Train and encode the tokenizer

```bash
python tokenizer/train_tokenizer.py
```

This creates or replaces the tokenizer merge table and encoded training
tensor.

### 4. Install dependencies

Create a virtual environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The notebook also uses `tqdm` and Plotly for progress and loss charts. If
they are not already available in the environment, install them with:

```bash
pip install tqdm plotly jupyter
```

### 5. Run the notebook

Start Jupyter from the repository root:

```bash
jupyter notebook main.ipynb
```

The notebook:

1. Loads `trained_tokenizer/train.bin`.
2. Splits it into 80% training and 20% validation tokens.
3. Builds `GPTDataset` examples with a context length of 256.
4. Uses batch size 256.
5. Instantiates `MoEGPT`.
6. Loads the initial state dictionary from `models/MoE_model`.
7. Optimizes with AdamW at learning rate `3e-4`.
8. Runs the training loop for one epoch by default.
9. Plots training and validation loss with Plotly.

The number of epochs, batch size, learning rate, and other settings can be
changed in the notebook.

### Device selection

The project selects the first available device in this order:

1. Apple Metal Performance Shaders (`mps`)
2. CUDA (`cuda`)
3. CPU (`cpu`)

This behavior is defined in [`deviceConfig.py`](./deviceConfig.py).

## Chat memory layer

The [`memory/`](./memory/) package is separate from model training. It
provides persistence primitives that can be connected to a future chat
interface:

- [`memory/database.py`](./memory/database.py) opens
  `memory/memory_store.sqlite3` and enables foreign keys.
- [`memory/conversation_store.py`](./memory/conversation_store.py) creates,
  reads, updates, and deletes conversations.
- [`memory/message_store.py`](./memory/message_store.py) stores messages with
  `user` or `assistant` roles and updates conversation timestamps.
- [`memory/memory_store.py`](./memory/memory_store.py) stores searchable,
  typed memories with importance, confidence, expiry, and active/archived
  status.
- [`memory/chat.py`](./memory/chat.py) contains a small interactive demo.

Run the demo from the repository root with:

```bash
python -m memory.chat
```

The demo accepts user input until `\exit`. Input beginning with
`remember:` is stored as a user memory. The demo currently prints a fixed
response supplied to `start_chat`; it does not generate responses from
`MoEGPT`.

## Repository layout

```text
.
├── dataDownload.py                  # Download the source datasets
├── dataset_concat_functions.py      # Build the unified text corpus
├── deviceConfig.py                  # Select MPS, CUDA, or CPU
├── main.ipynb                       # Notebook training workflow
├── requirements.txt                 # Core Python dependencies
├── dataset/
│   ├── input.txt                    # Tiny Shakespeare
│   ├── combined_corpus.txt          # Concatenated training text
│   ├── code_csv/train.csv          # Python code dataset export
│   └── arxiv_csv/arxiv.csv         # SciQ dataset export
├── tokenizer/
│   ├── BPT_tokenizer.py             # MinBPE implementation
│   └── train_tokenizer.py           # Train and serialize tokenizer
├── trained_tokenizer/
│   ├── moe_tokenizer.json           # Learned BPE merges
│   └── train.bin                    # Encoded token tensor
├── model_code/
│   ├── baseline_model.py             # Standard GPT experiment
│   ├── moe_gpt.py                   # Sparse MoE GPT
│   ├── router.py                    # Expert router
│   ├── experts.py                   # Expert feed-forward network
│   └── SparseMoEFeedForward.py      # Routing and balancing logic
├── loops_for_model/
│   └── training_loops.py             # Train/validation loop
├── models/
│   ├── MoE_model                    # MoE state dictionary
│   └── expert_model                 # Standalone expert state dictionary
└── memory/                           # SQLite conversations and memories
```

## Important notes and limitations

- The project currently has no automated test suite or standalone text
  generation command.
- The checked-in model files are PyTorch state dictionaries, not complete
  serialized model objects. Recreate the matching Python model class before
  loading them.
- Keep tokenizer and model vocabulary sizes synchronized. The MoE model
  expects 1,500 token IDs, while the baseline currently declares a different
  vocabulary size.
- `dataDownload.py` requires network access and can download several large
  files. Existing local files may be overwritten.
- The `arxiv_csv` name is historical; its current contents come from SciQ.
- The corpus builder intentionally uses only the SciQ `question` field, not
  the answer, distractors, or support text.
- The generated datasets, model checkpoints, caches, and SQLite database can
  be large. Check the repository ignore rules before committing regenerated
  artifacts.

## License and data attribution

This repository does not currently declare a project license. Before
redistributing the code or generated corpus, add an appropriate license and
review the licenses and terms of the upstream datasets and Tiny Shakespeare
source.