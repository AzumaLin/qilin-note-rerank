"""
Fine-tuned Qwen Reranker Inference + Evaluation.

Loads the fine-tuned Qwen3-Reranker-0.6B checkpoint (AutoModelForSequenceClassification),
scores candidates from search_result_details_with_idx, loads official qrels from
search.test.qrels.csv, then computes MRR/MAP/Recall/Precision @10 and @100.

Evaluated on queries [sample_start, sample_end) — default 0-1000 (same split as official baseline).
Relevance labels loaded from the official search.test.qrels.csv.

Arguments:
  --ckpt_mode   which fine-tuned checkpoint to load  (original | summary | combined)
  --infer_mode  how to build document text at inference (original | summary | combined)
                defaults to same as --ckpt_mode

Example — notecard-fine-tuned model, evaluated three ways:
    python -X utf8 qwen_reranker_eval.py --ckpt_mode summary --infer_mode original
    python -X utf8 qwen_reranker_eval.py --ckpt_mode summary --infer_mode summary
    python -X utf8 qwen_reranker_eval.py --ckpt_mode summary --infer_mode combined
"""

import os
import sys
import json
import argparse

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ========== 配置 ==========
def get_config(ckpt_mode: str, infer_mode: str) -> dict:
    return {
        'ckpt_mode':  ckpt_mode,
        'infer_mode': infer_mode,

        'model_path':   os.path.join(PROJECT_ROOT, 'output',
                                     f'A2qwen_ft_{ckpt_mode}'),

        'notes_dir':    os.path.join(PROJECT_ROOT, 'datasets', 'qilin', 'notes'),
        'test_path':    os.path.join(PROJECT_ROOT, 'datasets', 'qilin', 'search_test',
                                     'train-00000-of-00001.parquet'),
        'summary_path': os.path.join(PROJECT_ROOT, 'output', 'test_notecard.json'),
        'qrels_path':   os.path.join(PROJECT_ROOT, 'datasets', 'search.test.qrels.csv'),

        'sample_start': 0,
        'sample_end':   1000,

        'max_length':   384 if infer_mode == 'combined' else 256,
        'batch_size':   16,

        'output_dir': os.path.join(PROJECT_ROOT, 'output',
                                   f'A2qwen_eval_ckpt{ckpt_mode}_infer{infer_mode}'),
    }


# ========== 评估指标 ==========
def calculate_metrics(sorted_results, qrels, k_list):
    max_k = max(k_list)
    agg = {k: {"mrr": 0.0, "map_sum": 0.0, "recall": 0.0, "precision": 0.0} for k in k_list}
    valid = 0

    for qid, relevant in qrels.items():
        ranked = sorted_results.get(qid)
        if not ranked:
            continue
        valid += 1
        hits = [pid in relevant for pid in ranked[:max_k]]

        for k in k_list:
            hits_k = hits[:k]
            if any(hits_k):
                agg[k]["mrr"] += 1.0 / (hits_k.index(True) + 1)
            ap, correct = 0.0, 0
            for i, is_rel in enumerate(hits_k, start=1):
                if is_rel:
                    correct += 1
                    ap += correct / i
            if correct:
                agg[k]["map_sum"] += ap / min(len(relevant), k)
            agg[k]["recall"]    += sum(hits_k) / len(relevant)
            agg[k]["precision"] += sum(hits_k) / k

    if valid == 0:
        return {f"{m}@{k}": 0.0 for k in k_list for m in ["MRR", "MAP", "Recall", "Precision"]}

    out = {"_valid_queries": valid}
    for k in k_list:
        out[f"MRR@{k}"]       = agg[k]["mrr"]      / valid
        out[f"MAP@{k}"]       = agg[k]["map_sum"]   / valid
        out[f"Recall@{k}"]    = agg[k]["recall"]    / valid
        out[f"Precision@{k}"] = agg[k]["precision"] / valid
    return out


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_mode',  choices=['original', 'summary', 'combined'], required=True,
                        help='fine-tuned checkpoint to load')
    parser.add_argument('--infer_mode', choices=['original', 'summary', 'combined'], default=None,
                        help='document text mode at inference (default: same as ckpt_mode)')
    args = parser.parse_args()
    infer_mode = args.infer_mode or args.ckpt_mode

    config = get_config(args.ckpt_mode, infer_mode)
    os.makedirs(config['output_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"ckpt_mode:  {config['ckpt_mode']}")
    print(f"infer_mode: {config['infer_mode']}")
    print(f"Model:      {config['model_path']}")

    # ── 加载 notes corpus ────────────────────────────────────────────────────
    print('\n加载 notes corpus...')
    notes_df = pd.concat([
        pd.read_parquet(os.path.join(config['notes_dir'], f))
        for f in sorted(os.listdir(config['notes_dir']))
        if f.endswith('.parquet')
    ], ignore_index=True)
    print(f'  {len(notes_df):,} 条')

    # ── 加载 search_test ─────────────────────────────────────────────────────
    print('加载 search_test...')
    test_df = pd.read_parquet(config['test_path']).iloc[
        config['sample_start']:config['sample_end']
    ]
    print(f'  {len(test_df)} 条 queries  [{config["sample_start"]}:{config["sample_end"]}]')

    # ── 加载 summaries（可选）───────────────────────────────────────────────
    summaries: dict[int, str] = {}
    if config['infer_mode'] in ('summary', 'combined'):
        if os.path.exists(config['summary_path']):
            print('加载 summaries...')
            with open(config['summary_path'], 'r', encoding='utf-8') as f:
                raw = json.load(f)
            summaries = {int(k): str(v) for k, v in raw.items()}
            print(f'  {len(summaries):,} 条')
        else:
            print(f'  [警告] summaries 文件不存在，fallback 到原文: {config["summary_path"]}')

    def get_text(note_idx: int) -> str:
        row      = notes_df.iloc[note_idx]
        title    = str(row['note_title']   or '').strip()
        content  = str(row['note_content'] or '').strip()
        original = (title + '\n' + content).strip()
        card = summaries.get(note_idx, '')
        if config['infer_mode'] == 'summary':
            return card if card else original
        if config['infer_mode'] == 'combined':
            return (card + '\n' + original).strip() if card else original
        return original

    # ── 加载官方 qrels ───────────────────────────────────────────────────────
    print('加载官方 qrels...')
    qrels_df = pd.read_csv(config['qrels_path'])
    qrels: dict[int, set[int]] = {}
    for _, r in qrels_df.iterrows():
        qid = int(r['qid'])
        if config['sample_start'] <= qid < config['sample_end']:
            qrels.setdefault(qid, set()).add(int(r['pid']))
    print(f'  有 qrels 的 queries: {len(qrels)}')

    # ── 加载微调模型 ─────────────────────────────────────────────────────────
    model_path = config['model_path']
    if not os.path.isdir(model_path):
        print(f'\n[ERROR] 模型目录不存在: {model_path}')
        print('请先完成训练: python3 -X utf8 qwen_reranker_train.py '
              f'--mode {config["ckpt_mode"]}')
        sys.exit(1)

    print(f'\n加载微调模型: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'   # causal LM seq-cls 需要 right padding

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, trust_remote_code=True, local_files_only=True
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # ── 推理 ─────────────────────────────────────────────────────────────────
    print('\n开始推理...')
    sorted_results: dict[int, list[int]] = {}

    for qid, row in enumerate(tqdm(test_df.itertuples(), total=len(test_df), desc='Scoring')):
        query    = str(row.query)
        details  = row.search_result_details_with_idx
        note_ids = [int(d['note_idx']) for d in details]

        if not note_ids:
            continue

        all_scores = []
        for start in range(0, len(note_ids), config['batch_size']):
            batch_ids   = note_ids[start: start + config['batch_size']]
            batch_texts = [get_text(idx) for idx in batch_ids]

            encoded = tokenizer(
                [query] * len(batch_ids), batch_texts,
                padding=True, truncation=True,
                max_length=config['max_length'], return_tensors='pt',
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                scores = model(**encoded).logits.squeeze(-1).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        ranked = [nid for nid, _ in sorted(
            zip(note_ids, all_scores), key=lambda x: x[1], reverse=True
        )]
        sorted_results[qid] = ranked

    # ── 保存推理结果 ─────────────────────────────────────────────────────────
    results_path = os.path.join(config['output_dir'], 'rerank_results.csv')
    with open(results_path, 'w') as f:
        f.write('qid,pid,rank\n')
        for qid, pids in sorted_results.items():
            for rank, pid in enumerate(pids, start=1):
                f.write(f'{qid},{pid},{rank}\n')
    print(f'推理结果已保存: {results_path}')

    # ── 评估 ─────────────────────────────────────────────────────────────────
    metrics = calculate_metrics(sorted_results, qrels, [10, 100])

    print('\n' + '=' * 58)
    print(f'Qwen Reranker (fine-tuned)  ckpt={config["ckpt_mode"]}  '
          f'infer={config["infer_mode"]}  '
          f'queries={config["sample_start"]}-{config["sample_end"]}')
    print('=' * 58)
    for key, val in metrics.items():
        if not key.startswith('_'):
            print(f'  {key:<18} {val:.4f}')
    print(f'  valid queries:    {metrics.get("_valid_queries", 0)}')
    print('=' * 58)

    metrics_path = os.path.join(config['output_dir'], 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'ckpt_mode': config['ckpt_mode'], 'infer_mode': config['infer_mode'],
                   'metrics': metrics}, f,
                  ensure_ascii=False, indent=2)
    print(f'指标已保存: {metrics_path}')


if __name__ == '__main__':
    main()
