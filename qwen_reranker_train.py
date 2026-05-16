"""
Fine-tune Qwen Reranker (0.6B) cross-encoder on Qilin search_train (first 1000 queries).
800 train / 200 validation split.

与 bert_rerank_train.py 策略完全一致：
  - Positive: engagement 信号 (click/collect/like/comment/share > 0)
  - Negative: in-pool hard negatives (零 engagement)
  - Loss: pointwise BCEWithLogitsLoss
  - 保存 val_loss 最优 checkpoint

Qwen 相关改动：
  - AutoModelForSequenceClassification (num_labels=1)
  - 全量微调（0.6B，16GB 显卡可跑）
  - tokenizer padding_side='right'（causal LM seq-cls 需要找最后一个非 pad token）
  - 输入格式保持 tokenizer(query, doc) 双段式

Modes:
  original -- title + content
  summary  -- note_summaries.json (fallback to original if missing)
  combined -- summary + title + content (notecard + 原文拼接)

Usage:
    python -X utf8 qwen_reranker_train.py
    python -X utf8 qwen_reranker_train.py --mode combined
    python -X utf8 qwen_reranker_train.py --model_name Qwen/Qwen3-Reranker-0.6B --mode combined
"""

import os
import sys
import json
import random
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

POSITIVE_FIELDS = ('click', 'collect', 'like', 'comment', 'share')


# ========== 配置 ==========
def get_config(model_name: str, mode: str) -> dict:
    safe_name = model_name.replace('/', '_').replace('-', '_')
    return {
        'model_name': model_name,
        'mode': mode,

        'notes_dir':    os.path.join(PROJECT_ROOT, 'datasets', 'qilin', 'notes'),
        'train_path':    os.path.join(PROJECT_ROOT, 'datasets', 'qilin', 'search_train',
                                     'train-00000-of-00001.parquet'),
        'summary_path': os.path.join(PROJECT_ROOT, 'output', 'train_notecard.json'),

        'n_queries':    5000,
        'train_size':   4000,
        'val_size':     1000,

        'max_length':   384 if mode == 'combined' else 256,
        'batch_size':   8,
        'epochs':       5,
        'lr':           2e-5,
        'warmup_ratio': 0.1,
        'neg_per_pos':  2,
        'seed':         42,

        'output_dir': os.path.join(PROJECT_ROOT, 'output', f'C2qwen_ft_{mode}'),
    }


# ========== 数据集 ==========
class RerankDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # [(query, note_text, label), ...]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer, max_length):
    queries    = [item[0] for item in batch]
    note_texts = [item[1] for item in batch]
    labels     = torch.tensor([item[2] for item in batch], dtype=torch.float)
    encoded = tokenizer(
        queries, note_texts,
        padding=True, truncation=True,
        max_length=max_length, return_tensors='pt',
    )
    return encoded, labels


# ========== 构建训练对 ==========
def build_pairs(rows, notes_df, summaries, mode, neg_per_pos):
    pairs = []

    def get_text(note_idx):
        row = notes_df.iloc[note_idx]
        title    = str(row['note_title']   or '').strip()
        content  = str(row['note_content'] or '').strip()
        original = (title + '\n' + content).strip()
        if mode == 'summary':
            return summaries.get(note_idx, original)
        if mode == 'combined':
            card = summaries.get(note_idx, '')
            return (card + '\n' + original).strip() if card else original
        return original

    for row in rows:
        query   = str(row['query'])
        details = row['search_result_details_with_idx']

        positives, negatives = [], []
        for d in details:
            note_idx = int(d['note_idx'])
            is_pos = any(float(d.get(f, 0) or 0) > 0 for f in POSITIVE_FIELDS)
            (positives if is_pos else negatives).append(note_idx)

        if not positives or not negatives:
            continue

        random.shuffle(negatives)
        for pos_idx in positives:
            pairs.append((query, get_text(pos_idx), 1))
            for neg_idx in negatives[:neg_per_pos]:
                pairs.append((query, get_text(neg_idx), 0))

    return pairs


# ========== 训练 / 验证一个 epoch ==========
def run_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        pbar = tqdm(dataloader, desc='Train' if train else 'Val', leave=False)
        for encoded, labels in pbar:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels  = labels.to(device)

            outputs = model(**encoded)
            loss    = loss_fn(outputs.logits.squeeze(-1), labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Qwen/Qwen3-Reranker-0.6B',
                        help='HuggingFace model ID 或本地路径')
    parser.add_argument('--mode', choices=['original', 'summary', 'combined'], default='original')
    args = parser.parse_args()

    config = get_config(args.model_name, args.mode)
    os.makedirs(config['output_dir'], exist_ok=True)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"Model:      {config['model_name']}")
    print(f"Mode:       {config['mode']}")
    print(f"Output dir: {config['output_dir']}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print('\n加载 notes corpus...')
    notes_dfs = [
        pd.read_parquet(os.path.join(config['notes_dir'], f))
        for f in sorted(os.listdir(config['notes_dir']))
        if f.endswith('.parquet')
    ]
    notes_df = pd.concat(notes_dfs, ignore_index=True)
    print(f'  {len(notes_df):,} 条')

    print('加载 search_train (前1000条)...')
    train_df = pd.read_parquet(config['train_path']).head(config['n_queries'])
    rows = train_df.to_dict('records')

    train_rows = rows[:config['train_size']]
    val_rows   = rows[config['train_size']:]
    print(f'  训练: {len(train_rows)} 条 | 验证: {len(val_rows)} 条')

    summaries: dict[int, str] = {}
    if config['mode'] in ('summary', 'combined'):
        if os.path.exists(config['summary_path']):
            print('加载 summaries...')
            with open(config['summary_path'], 'r', encoding='utf-8') as f:
                raw = json.load(f)
            summaries = {int(k): str(v) for k, v in raw.items()}
            print(f'  {len(summaries):,} 条')
        else:
            print(f'  [警告] summaries 文件不存在，fallback 到原文: {config["summary_path"]}')

    # ── 构建训练对 ─────────────────────────────────────────────────────────────
    print('\n构建训练对...')
    train_pairs = build_pairs(train_rows, notes_df, summaries, config['mode'], config['neg_per_pos'])
    val_pairs   = build_pairs(val_rows,   notes_df, summaries, config['mode'], config['neg_per_pos'])
    random.shuffle(train_pairs)
    print(f'  训练对: {len(train_pairs):,}  '
          f'(正:{sum(p[2] for p in train_pairs):,} '
          f'负:{sum(1-p[2] for p in train_pairs):,})')
    print(f'  验证对: {len(val_pairs):,}')

    # ── 加载模型 + LoRA ────────────────────────────────────────────────────────
    print(f'\n加载模型: {config["model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # causal LM 做 seq-cls 时要 right padding，模型才能找到最后一个真实 token
    tokenizer.padding_side = 'right'

    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=1,
        trust_remote_code=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── DataLoader ────────────────────────────────────────────────────────────
    fn = lambda b: collate_fn(b, tokenizer, config['max_length'])
    train_loader = DataLoader(RerankDataset(train_pairs), batch_size=config['batch_size'],
                              shuffle=True,  collate_fn=fn)
    val_loader   = DataLoader(RerankDataset(val_pairs),   batch_size=config['batch_size'],
                              shuffle=False, collate_fn=fn)

    # ── 优化器 + 调度器 ───────────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    total_steps  = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.BCEWithLogitsLoss()

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f'\n开始训练 ({config["epochs"]} epochs)...\n')
    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler,
                               loss_fn, device, train=True)
        val_loss   = run_epoch(model, val_loader, optimizer, scheduler,
                               loss_fn, device, train=False)

        print(f'Epoch {epoch}/{config["epochs"]}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            model.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            print(f'  ✓ 最佳模型已保存 (val_loss={best_val_loss:.4f})')

    print(f'\n训练完成。最佳 epoch: {best_epoch}，val_loss: {best_val_loss:.4f}')
    print(f'模型保存在: {config["output_dir"]}')


if __name__ == '__main__':
    main()
