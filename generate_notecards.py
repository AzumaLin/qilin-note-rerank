"""
Generate LLM-based notecards for Qilin notes.

Reads candidate note indices from search_train or search_test queries,
denoises note text, summarizes with Qwen3-4B-Instruct, and saves to JSON.

Output format (one entry per note_idx):
  类目: <category>
  关键词: <k1, k2, k3, k4, k5>
  场景: <search scenario, ≤10 chars>
  人群: <target audience>
  实体: <brands / products / people / locations, or 无>
  摘要: <one-sentence search-friendly description>

Usage:
    # Generate notecards for training data
    python generate_notecards.py --split train --output data/train_notecard.json

    # Generate notecards for test data
    python generate_notecards.py --split test --output data/test_notecard.json
"""

import os
import re
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Regex denoising ────────────────────────────────────────────────────────────

_RE_HASHTAG         = re.compile(r'#[^#\[\]]*\[话题\]#')
_RE_PLATFORM_MARKER = re.compile(r'\[[^\[\]]*R\]')
_RE_EMOJI           = re.compile(
    '['
    '\U0001F600-\U0001F64F'
    '\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF'
    '\U0001F1E0-\U0001F1FF'
    '\U00002702-\U000027B0'
    '\U0000FE00-\U0000FE0F'
    '\U0001F900-\U0001F9FF'
    '\U0001FA00-\U0001FA6F'
    '\U0001FA70-\U0001FAFF'
    '\U00002600-\U000026FF'
    '\U0000200D'
    '\U00002B50'
    '\U000023F0-\U000023FA'
    '\U0000203C-\U00003299'
    ']+',
    flags=re.UNICODE
)
_RE_WHITESPACE = re.compile(r'\s+')


def regex_denoise(text: str) -> str:
    text = _RE_HASHTAG.sub('', text)
    text = _RE_PLATFORM_MARKER.sub('', text)
    text = _RE_EMOJI.sub('', text)
    return _RE_WHITESPACE.sub(' ', text).strip()


# ── LLM prompt ────────────────────────────────────────────────────────────────

def build_prompt(title: str, content: str) -> str:
    return (
        "你是搜索摘要助手。将小红书帖子总结成结构化卡片，用于搜索匹配。\n"
        "严格按照以下格式输出，不要输出任何其他内容：\n"
        "类目: <类目>\n"
        "关键词: <k1, k2, k3, k4, k5>\n"
        "场景: <用户在什么情况下会搜索这个，10字以内>\n"
        "人群: <目标人群>\n"
        "实体: <品牌、产品、人物、地点>\n"
        "摘要: <一句话，适合搜索匹配的简洁描述>\n\n"
        "规则：\n"
        "1. 关键词：混合类目词、属性词、用户实际会搜索的意图词，共5个\n"
        "2. 场景：描述用户搜索场景，10字以内（例：小个子找冬季显瘦穿搭）\n"
        "3. 实体：只提取标题、正文、标签中明确出现的品牌/产品/人物/地点，没有则写「无」\n"
        "4. 摘要：用简洁陈述句改写，去除网络用语和营销夸张词\n\n"
        "示例：\n"
        "Title: 蕉内羽绒服穿搭显瘦又保暖\n"
        "Body: 3件都是蕉内氢气羽绒服502Cloud，实穿又保暖，不怕臃肿，短款小个子友好\n"
        "类目: 穿搭\n"
        "关键词: 羽绒服, 短款, 显瘦, 保暖, 小个子\n"
        "场景: 小个子找冬季显瘦保暖穿搭\n"
        "人群: 小个子女生, 冬季穿搭爱好者\n"
        "实体: 蕉内, 氢气羽绒服502Cloud\n"
        "摘要: 蕉内短款羽绒服穿搭，显瘦保暖，小个子友好\n\n"
        f"Title: {title}\n"
        f"Body: {content}\n"
    )


def summarize_batch(prompts, tokenizer, model, device, max_new_tokens):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    inputs = tokenizer(
        texts, return_tensors='pt', padding=True,
        truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i, output in enumerate(outputs):
        generated = output[inputs['input_ids'][i].shape[0]:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',      choices=['train', 'test'], default='train',
                        help='Which query split to collect candidate notes from')
    parser.add_argument('--output',     default=None,
                        help='Output JSON path (default: data/train_notecard.json or data/test_notecard.json)')
    parser.add_argument('--llm_model',  default='Qwen/Qwen3-4B-Instruct',
                        help='LLM for summarization')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=350)
    parser.add_argument('--rerank_depth',   type=int, default=100,
                        help='Number of top candidates per query to collect')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(DATA_DIR, f'{args.split}_notecard.json')

    query_path = os.path.join(DATA_DIR, f'search_{args.split}', 'train-00000-of-00001.parquet')
    notes_dir  = os.path.join(DATA_DIR, 'notes')

    print(f"Split:      {args.split}")
    print(f"Query file: {query_path}")
    print(f"Output:     {args.output}")

    # 1. Load queries and collect candidate note indices
    print('\n[1/4] Loading queries...')
    query_df = pd.read_parquet(query_path)
    print(f'  {len(query_df):,} queries')

    note_idxs = set()
    for row in query_df.itertuples():
        details = row.search_result_details_with_idx
        for d in details[:args.rerank_depth]:
            note_idxs.add(int(d['note_idx']))
    print(f'  Unique candidate notes: {len(note_idxs):,}')

    # 2. Load notes corpus
    print('\n[2/4] Loading notes corpus...')
    notes_df = pd.concat([
        pd.read_parquet(os.path.join(notes_dir, f))
        for f in sorted(os.listdir(notes_dir))
        if f.endswith('.parquet')
    ], ignore_index=True)
    print(f'  {len(notes_df):,} notes')

    # 3. Denoise and prepare prompts
    print('\n[3/4] Denoising notes...')
    idx_list = sorted(note_idxs)
    prompts  = []
    for note_idx in tqdm(idx_list, desc='Denoising'):
        row     = notes_df.iloc[note_idx]
        title   = regex_denoise(str(row['note_title']   or ''))
        content = regex_denoise(str(row['note_content'] or ''))
        prompts.append(build_prompt(title, content))

    # 4. LLM summarization
    print(f'\n[4/4] Loading LLM: {args.llm_model}...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model, trust_remote_code=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    print(f'  Summarizing {len(prompts):,} notes (batch_size={args.batch_size})...')
    summaries = {}
    for start in tqdm(range(0, len(prompts), args.batch_size), desc='Summarizing'):
        end          = min(start + args.batch_size, len(prompts))
        batch_out    = summarize_batch(prompts[start:end], tokenizer, model, device, args.max_new_tokens)
        for note_idx, summary in zip(idx_list[start:end], batch_out):
            summaries[note_idx] = summary

    del model, tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in summaries.items()}, f, ensure_ascii=False, indent=2)

    print(f'\n完成！保存了 {len(summaries):,} 条 notecards → {args.output}')


if __name__ == '__main__':
    main()
