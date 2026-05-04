

# Qilin Note Rerank
## Note Summarization Pipeline

Pre-processing pipeline that denoises and summarizes candidate notes using an LLM, producing structured cards for downstream reranking.

### Files

| File | Description |
|------|-------------|
| `main/summarize_notes.py` | Main pipeline script |
| `main/output/test_notecard.json` | Pre-computed note summaries for the full test set |

### Output Format

`test_notecard.json` maps `note_id (str)` → structured summary string with six fields:

```
类目: <category>
关键词: <k1, k2, k3, k4, k5>
场景: <search scenario, ≤10 chars>
人群: <target audience>
实体: <brands / products / people / locations, or 无>
摘要: <one-sentence search-friendly description>
```

**Coverage:** 112,595 unique notes drawn from top-100 DPR candidates across all test queries (~5.7% of the 1.98M note corpus).

### How to Run

```bash
conda activate qilin
cd main
python3 summarize_notes.py
```

Key parameters in the `CONFIG` block at the top of the script:

| Parameter | Description |
|-----------|-------------|
| `sample_start` / `sample_end` | Query index range to process (`None` = all) |
| `llm_batch_size` | Batch size for LLM inference (4 for 16GB GPU, 32–64 for A100) |
| `output_path` | Path to save the output JSON |

To run on a SLURM cluster:

```bash
sbatch main/submit1.sh
tail -f logs/summarize_<jobid>.log
```

### Pipeline Steps

1. Load `search_test` queries and `notes` corpus from the Qilin dataset
2. Collect unique candidate note indices from top-100 DPR results
3. Regex denoise note titles and content (remove hashtags, platform markers, emoji)
4. Batch-summarize with `Qwen/Qwen3-4B-Instruct` into structured cards
5. Save to JSON
