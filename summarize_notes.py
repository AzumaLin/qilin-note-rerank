"""
Note Summarization Pipeline for Qilin BERT Rerank
Standalone script that pre-processes notes: regex denoising + LLM summarization.
Generates output/note_summaries.json for use by bert_inference_with_summary.py.

Usage:
    python -X utf8 summarize_notes.py
"""

import os
import re
import json

# Force HuggingFace to use local cache only — prevents downloading when cache exists.
# Remove or set to '0' only if you intentionally want to download.
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

print("[boot] importing torch...", flush=True)
import torch
print("[boot] importing tqdm...", flush=True)
from tqdm import tqdm
print("[boot] importing datasets...", flush=True)
from datasets import load_dataset
print("[boot] importing transformers...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
print("[boot] all imports done.", flush=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== Config ==========
CONFIG = {
    'dataset_name_or_path': 'THUIR/Qilin',
    'test_data_key': 'search_test',
    'results_key': 'search_results',
    'rerank_depth': 100,

    # Take queries [sample_start, sample_end) from the front of the dataset.
    # Set sample_end to None to use all remaining queries after sample_start.
    'sample_start': 0,
    'sample_end': 3000,

    # Local HuggingFace datasets cache directory on the cluster.
    # Set to None to use the default (~/.cache/huggingface/datasets).
    # Example: '/data/shared/hf_cache' or '/scratch/datasets'
    'hf_cache_dir': None,

    # LLM for summarization
    'llm_model': 'Qwen/Qwen3-4B-Instruct-2507',
    'llm_max_new_tokens': 350,
    'llm_batch_size': 4,

    'output_path': os.path.join(PROJECT_ROOT, 'output', 'test_notecard1.json'),
}


# ========== Regex Denoising ==========

# Pattern: #话题文本[话题]#
_RE_HASHTAG = re.compile(r'#[^#\[\]]*\[话题\]#')
# Pattern: [笑哭R] style platform markers — [任意文字R]
_RE_PLATFORM_MARKER = re.compile(r'\[[^\[\]]*R\]')
# Unicode emoji ranges (covers most common emoji blocks)
_RE_EMOJI = re.compile(
    '['
    '\U0001F600-\U0001F64F'  # emoticons
    '\U0001F300-\U0001F5FF'  # symbols & pictographs
    '\U0001F680-\U0001F6FF'  # transport & map
    '\U0001F1E0-\U0001F1FF'  # flags
    '\U00002702-\U000027B0'  # dingbats
    '\U0000FE00-\U0000FE0F'  # variation selectors
    '\U0001F900-\U0001F9FF'  # supplemental symbols
    '\U0001FA00-\U0001FA6F'  # chess symbols
    '\U0001FA70-\U0001FAFF'  # symbols extended-A
    '\U00002600-\U000026FF'  # misc symbols
    '\U0000200D'             # zero width joiner
    '\U00002B50'             # star
    '\U000023F0-\U000023FA'  # misc technical
    '\U0000203C-\U00003299'  # enclosed CJK
    ']+',
    flags=re.UNICODE
)
# Collapse multiple whitespace into single space
_RE_WHITESPACE = re.compile(r'\s+')


def load_dataset_cached(name, split, cache_dir=None):
    """Load HuggingFace dataset, preferring local cache.

    Pass cache_dir to point at a non-default HF cache location (e.g. on a cluster).
    Falls back to downloading only when the local cache is genuinely missing.
    """
    kwargs = {}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    try:
        return load_dataset(name, split, **kwargs)['train']
    except Exception as e:
        print(f"  [load error] {e}")
        raise


def load_pretrained_cached(cls, model_id, **kwargs):
    """Load HuggingFace model/tokenizer, preferring local cache."""
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError:
        return cls.from_pretrained(model_id, **kwargs)


def regex_denoise(text: str) -> str:
    """Apply regex-based denoising to remove hashtags, platform markers, emojis."""
    text = _RE_HASHTAG.sub('', text)
    text = _RE_PLATFORM_MARKER.sub('', text)
    text = _RE_EMOJI.sub('', text)
    text = _RE_WHITESPACE.sub(' ', text).strip()
    return text


# ========== Main Pipeline ==========

def collect_candidate_note_idxs(test_data, results_key, rerank_depth):
    """Collect all unique candidate note_idx from test queries."""
    note_idxs = set()
    for item in test_data:
        candidates = item[results_key]
        if isinstance(candidates[0], int):
            candidates = [[x, 0.0] for x in candidates]
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:rerank_depth]
        for candidate in candidates:
            note_idxs.add(int(candidate[0]))
    return note_idxs


# def build_llm_prompt(title: str, content: str) -> str:
#     return (
#         "You are a search summarization assistant. Summarize a Xiaohongshu post into a structured card for search matching.\n"
#         "Output EXACTLY this format, no extra text:\n"
#         "Category: <category>\n"
#         "Keywords: <k1, k2, k3, k4, k5>\n"
#         "Audience: <target audience>\n"
#         "Entities: <brands, products, people, locations>\n"
#         "Gist: <one sentence summary optimized for search>\n\n"
#         "Rules:\n"
#         "1. Keywords: mix of category words, attribute words, and intent words users would actually search\n"
#         "2. Entities: only specific named things (brands, products, celebrities, places), write 'None' if absent\n"
#         "3. Audience: who this post targets (e.g. petite girls, new parents, fitness beginners)\n"
#         "4. Gist: rewrite in plain search-friendly language, remove slang and marketing buzzwords\n"
#         "5. If post has no useful content, output 'Category: None' and leave other fields as 'None'\n\n"
#         "Example 1:\n"
#         "Title: 蕉内羽绒服穿搭显瘦又保暖\n"
#         "Body: 3件都是蕉内氢气羽绒服502Cloud，实穿又保暖，不怕臃肿，短款小个子友好\n"
#         "Category: 穿搭\n"
#         "Keywords: 羽绒服, 短款, 显瘦, 保暖, 小个子\n"
#         "Audience: 小个子女生, 冬季穿搭爱好者\n"
#         "Entities: 蕉内, 氢气羽绒服502Cloud\n"
#         "Gist: 蕉内短款羽绒服穿搭，显瘦保暖，小个子友好\n\n"
#         "Example 2:\n"
#         "Title: 必胜客芝士和牛至尊堡\n"
#         "Body: 有这么好吃的芝士和牛至尊堡现在才说！这么一大块和牛肉片，搭配火候刚刚好的煎蛋，一口下去简直杀疯我\n"
#         "Category: 美食\n"
#         "Keywords: 汉堡, 芝士堡, 和牛, 测评, 快餐\n"
#         "Audience: 美食爱好者, 汉堡爱好者\n"
#         "Entities: 必胜客, 芝士和牛至尊堡\n"
#         "Gist: 必胜客芝士和牛至尊堡测评，和牛肉片搭配煎蛋，口感好\n\n"
#         "Example 3:\n"
#         "Title: 年少不知富婆好\n"
#         "Body: 素材来源于网络，如有侵权请联系删除\n"
#         "Category: None\n"
#         "Keywords: None\n"
#         "Audience: None\n"
#         "Entities: None\n"
#         "Gist: None\n\n"
#         f"Title: {title}\n"
#         f"Body: {content}\n"
#     )


def build_llm_prompt(title: str, content: str) -> str:
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
        "4. 摘要：用简洁陈述句改写，去除网络用语和营销夸张词\n"
        "5. 标签（#...#）：提取地点标签、活动标签、品牌标签作为实体或关键词，忽略#日常分享# #笔记灵感#等无意义标签\n\n"
        "示例1：\n"
        "Title: 蕉内羽绒服穿搭显瘦又保暖\n"
        "Body: 3件都是蕉内氢气羽绒服502Cloud，实穿又保暖，不怕臃肿，短款小个子友好\n"
        "类目: 穿搭\n"
        "关键词: 羽绒服, 短款, 显瘦, 保暖, 小个子, 冬季穿搭, 蕉内, 不臃肿\n"
        "场景: 小个子找冬季显瘦保暖穿搭\n"
        "人群: 小个子女生, 冬季穿搭爱好者\n"
        "实体: 蕉内, 氢气羽绒服502Cloud\n"
        "摘要: 蕉内短款羽绒服穿搭，显瘦保暖，小个子友好\n\n"
        "示例2：\n"
        "Title: nan\n"
        "Body: 一分钟教会你国补和省补的区别 #买车那点事儿# #新手买车攻略# #新能源汽车补贴# #买车推荐#\n"
        "类目: 汽车\n"
        "关键词: 国补, 省补, 新能源补贴, 买车攻略, 购车补贴, 新手买车, 补贴区别, 新能源汽车\n"
        "场景: 买新能源车想了解国补省补怎么算\n"
        "人群: 准备购车的新手, 新能源车买家\n"
        "实体: 无\n"
        "摘要: 一分钟了解国补和省补的区别，新手买车新能源补贴攻略\n\n"
        "示例3：\n"
        "Title: 追星女行李托运保姆级教程‼️千万不要被扔啊\n"
        "Body: #追星女行李托运教程# #时代少年团五周年# #追星女必看# #时代少年团重庆五周年演唱会#\n"
        "类目: 演唱会\n"
        "关键词: 追星, 行李托运, 演唱会, 保姆级教程, 时代少年团, 重庆演唱会, 五周年, 行李不丢\n"
        "场景: 追星去演唱会想知道行李怎么托运\n"
        "人群: 追星女孩, 时代少年团粉丝\n"
        "实体: 时代少年团, 时代少年团重庆五周年演唱会\n"
        "摘要: 追星女演唱会行李托运保姆级教程，时代少年团重庆五周年演唱会适用\n\n"
        "示例4：\n"
        "Title: 在南京也可以吃到内蒙古奶疙瘩噜～\n"
        "Body: #南京国际博览中心# #南京中外商品博览会# #内蒙古奶疙瘩# #乌梨# #文玩葡萄# #糕点软软糯糯# #日常分享#\n"
        "类目: 美食\n"
        "关键词: 奶疙瘩, 内蒙古特产, 南京, 博览会, 特色食品, 南京美食, 乌梨, 糕点\n"
        "场景: 在南京想找内蒙古特产或博览会美食\n"
        "人群: 对内蒙古特产感兴趣的人, 南京本地人\n"
        "实体: 南京国际博览中心, 南京中外商品博览会\n"
        "摘要: 南京中外商品博览会可以品尝到内蒙古奶疙瘩、乌梨等特产糕点\n\n"
        "示例5：\n"
        "Title: nan\n"
        "Body: 我只是想同步个微信步数而已，不是让你记录我运动，立马把手环取消绑定 #华为手环9# #华为运动健康#\n"
        "类目: 数码\n"
        "关键词: 华为手环, 微信步数, 手环绑定, 运动记录, 取消绑定, 华为手环9, 步数同步, 隐私\n"
        "场景: 华为手环同步微信步数出现问题想解决\n"
        "人群: 华为手环用户, 关注隐私的用户\n"
        "实体: 华为手环9, 华为运动健康\n"
        "摘要: 华为手环9同步微信步数时会记录运动数据，可取消绑定关闭\n\n"
        f"Title: {title}\n"
        f"Body: {content}\n"
    )

def summarize_batch(prompts, tokenizer, model, device, max_new_tokens):
    """Run LLM summarization on a batch of prompts."""
    messages_batch = [
        [{"role": "user", "content": p}] for p in prompts
    ]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (strip input tokens)
    summaries = []
    for i, output in enumerate(outputs):
        input_len = inputs['input_ids'][i].shape[0]
        generated = output[input_len:]
        summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
        summaries.append(summary)
    return summaries


def main():
    print("=" * 60)
    print("Note Summarization Pipeline")
    print("=" * 60)

    # 1. Load data (local cache preferred)
    print("\n[1/5] Loading test data and corpus...")
    cache_dir = CONFIG['hf_cache_dir']
    test_data = load_dataset_cached(CONFIG['dataset_name_or_path'], CONFIG['test_data_key'], cache_dir)
    start = CONFIG['sample_start']
    end = CONFIG['sample_end'] if CONFIG['sample_end'] is not None else len(test_data)
    end = min(end, len(test_data))
    test_data = test_data.select(range(start, end))
    corpus = load_dataset_cached(CONFIG['dataset_name_or_path'], 'notes', cache_dir)
    print(f"  Test queries [{start}:{end}]: {len(test_data)}, Corpus size: {len(corpus)}")

    # 2. Collect unique candidate note_idx
    print("\n[2/5] Collecting unique candidate note indices...")
    note_idxs = collect_candidate_note_idxs(
        test_data, CONFIG['results_key'], CONFIG['rerank_depth']
    )
    print(f"  Unique candidate notes: {len(note_idxs)}")

    # 3. Regex denoise all notes
    print("\n[3/5] Regex denoising all candidate notes...")
    notes_for_llm = {}  # note_idx -> (cleaned_title, cleaned_content)

    for note_idx in tqdm(note_idxs, desc="Denoising"):
        note = corpus[note_idx]
        title = note['note_title'] or ''
        content = note['note_content'] or ''
        notes_for_llm[note_idx] = (regex_denoise(title), regex_denoise(content))

    print(f"  Notes to summarize: {len(notes_for_llm)}")

    # 4. LLM summarization for all notes
    summaries = {}

    if notes_for_llm:
        print(f"\n[4/5] Loading LLM: {CONFIG['llm_model']}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        tokenizer = load_pretrained_cached(AutoTokenizer, CONFIG['llm_model'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        model = load_pretrained_cached(
            AutoModelForCausalLM,
            CONFIG['llm_model'],
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()

        # Build prompts for all notes
        all_idx_list = list(notes_for_llm.keys())
        prompts = [
            build_llm_prompt(notes_for_llm[idx][0], notes_for_llm[idx][1])
            for idx in all_idx_list
        ]

        print(f"  Summarizing {len(prompts)} notes in batches of {CONFIG['llm_batch_size']}...")
        batch_size = CONFIG['llm_batch_size']

        for start in tqdm(range(0, len(prompts), batch_size), desc="LLM Summarizing"):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_idxs = all_idx_list[start:end]

            batch_summaries = summarize_batch(
                batch_prompts, tokenizer, model, device,
                CONFIG['llm_max_new_tokens']
            )

            for idx, summary in zip(batch_idxs, batch_summaries):
                summaries[idx] = summary

        # Free GPU memory
        del model
        del tokenizer
        if device == 'cuda':
            torch.cuda.empty_cache()
    else:
        print("\n[4/5] No notes to summarize.")

    # 5. Save to JSON
    print(f"\n[5/5] Saving summaries to {CONFIG['output_path']}...")
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)

    # JSON keys must be strings
    output = {str(k): v for k, v in summaries.items()}
    with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(output)} note summaries.")
    print(f"  Output: {CONFIG['output_path']}")

    # Quick stats
    lengths = [len(v) for v in output.values()]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"  Avg summary length: {avg_len:.1f} chars")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")


if __name__ == '__main__':
    main()
