# Qwen3-Reranker Fine-tuning on Qilin

Fine-tune [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) on the [Qilin dataset](https://huggingface.co/datasets/NTCIR-18-Qilin/Qilin) for Chinese social media search reranking.

Uses SFT (cross-entropy on yes/no token) with any-engagement positives.

## Data Setup

Download the Qilin dataset from HuggingFace and place files as follows:

```
reranker/data/
├── notes/
│   ├── train-00000-of-00005.parquet
│   ├── train-00001-of-00005.parquet
│   ├── train-00002-of-00005.parquet
│   ├── train-00003-of-00005.parquet
│   └── train-00004-of-00005.parquet
├── search_train/
│   └── train-00000-of-00001.parquet
├── search_test/
│   └── train-00000-of-00001.parquet
├── qrels/
│   └── search.test.qrels.csv
├── train_notecard.json          ← LLM-generated notecards for search_train (optional)
└── test_notecard.json           ← LLM-generated notecards for search_test  (optional)
```

Download commands:

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="NTCIR-18-Qilin/Qilin",
    repo_type="dataset",
    local_dir="data_raw"
)
EOF
```

Then move the relevant files into `data/` as shown above. The `search.test.qrels.csv` file is the official relevance judgement file from the Qilin task.

Notecards (`train_notecard.json` / `test_notecard.json`) are optional — only needed for `--mode summary` or `--mode combined`. Generate them with `generate_notecards.py` (see below).

## Generating Notecards (optional)

Notecards are LLM-generated structured summaries of notes, used for `--mode summary/combined`.

```bash
# Generate notecards for training queries (saves to data/train_notecard.json)
python generate_notecards.py --split train

# Generate notecards for test queries (saves to data/test_notecard.json)
python generate_notecards.py --split test

# Custom options
python generate_notecards.py --split train --batch_size 8 --llm_model Qwen/Qwen3-4B-Instruct
```

Requires a GPU with sufficient VRAM (≥16GB recommended for batch_size=4).

## Training

```bash
# Train on original note text
python train.py --mode original --prefix run1

# Train on LLM-generated notecards (requires train_notecard.json)
python train.py --mode summary --prefix run1

# Train on notecard + original text combined
python train.py --mode combined --prefix run1

# Custom learning rate
python train.py --mode summary --prefix run1 --lr 5e-6
```

Checkpoints are saved to `output/{prefix}_qwen3_ft_{mode}/`.

Key hyperparameters (edit `get_config` in `train.py` to change):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Per-GPU batch size (effective batch = 4 with accum_steps=2) |
| `lr` | 1e-5 | Learning rate (override with `--lr`) |
| `epochs` | 3 | Number of epochs |
| `neg_per_pos` | 2 | Negatives per positive sample |
| `max_length` | 256 | Max token length (384 for combined mode) |

## Evaluation

```bash
# Evaluate a checkpoint with original text at inference
python eval.py --ckpt_dir output/run1_qwen3_ft_summary --infer_mode original

# Evaluate with summary text at inference
python eval.py --ckpt_dir output/run1_qwen3_ft_summary --infer_mode summary
```

Results (MRR@10, MAP@10, Recall@10, Precision@10) are saved to `output/{ckpt_dir}/eval_infer{mode}/metrics.json`.

## Requirements

```
torch
transformers
pandas
pyarrow
tqdm
```
