
<div align="center">
<img src="pics/icon.jpg" alt="Qilin Logo" width="200"/>
</div>

# Qilin

Qilin is a large-scale multimodal dataset designed for advancing research in search, recommendation, and Retrieval-Augmented Generation (RAG) systems. This repository contains the official implementation of the dataset paper, baseline models, and evaluation tools.

## 📢 News
**[2025-04-05]** 🎉 [Qilin](https://arxiv.org/abs/2503.00501) is accepted to SIGIR 2025! Huge thanks to all collaborators!

**[2024-03-27]** 🚀 Qilin surpasses 1,000 downloads on [HuggingFace](https://huggingface.co/datasets/THUIR/qilin)! Thank you for your support!

**[2024-03-18]** 🔥Image resources are now available for download through [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/af72ab5dbba1460da6c0/)! 

## Dataset Overview

Qilin provides comprehensive data for three main scenarios:

### Search Dataset
- Training set: 44,024 samples
- Testing set: 6,192 samples
- Features:
  - Rich query metadata
  - User interaction logs
  - Ground clicked labels

### Recommendation Dataset
- Training set: 83,437 samples
- Testing set: 11,115 samples
- Features:
  - Detailed user interaction history
  - Candidate note pools
  - Contextual features
  - Ground clicked labels

### Key Characteristics
- Multiple content modalities (text, images, video thumbnails)
- Rich user interaction data
- Comprehensive evaluation metrics
- Support for RAG system development

## Repository Structure

- `baselines/`: Implementation of state-of-the-art baseline models
- `datasets/`: Dataset files and processing scripts
  - `toy_data/`: Small sample dataset for quick exploration
  - `qilin/`: Complete dataset (after downloading)

## Getting Started

### Installation

```bash
pip install -r baselines/requirements.txt
```

### Data and Model Preparation

1. Download the Qilin dataset from [Hugging Face](https://huggingface.co/datasets/THUIR/qilin)
2. Extract and place the dataset in the `datasets/qilin` directory
3. Download the required models:
   - [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
   - [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
   - [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
4. Place the downloaded models in the `model` directory

## Citation

If you use Qilin in your research, please cite our paper:

```
@misc{chen2025qilinmultimodalinformationretrieval,
      title={Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions}, 
      author={Jia Chen and Qian Dong and Haitao Li and Xiaohui He and Yan Gao and Shaosheng Cao and Yi Wu and Ping Yang and Chen Xu and Yao Hu and Qingyao Ai and Yiqun Liu},
      year={2025},
      eprint={2503.00501},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2503.00501}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

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
