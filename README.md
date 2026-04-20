
# FLiP: Factorized Linear Projection for Interpreting Multimodal Multilingual Sentence Embeddings

FLiP is a diagnostic tool for interpreting pretrained sentence embedding spaces. It trains a factorized log-linear model to recover lexical content (keywords) from sentence embeddings via a single linear projection — no fine-tuning of the encoder, no heuristics.

Under review:
> Santosh Kesiraju, Bolaji Yusuf, Simon Sedlacek, Oldrich Plchot, Petr Schwarz.
> *FLiP: Towards understanding and interpreting multimodal multilingual sentence embeddings.*
> Speech@FIT, Brno University of Technology.

[Read the paper](paper/FLiP.pdf)

---

## Main results

Keyword extraction accuracy (fraction of in-vocabulary reference tokens recovered) on Mozilla Common Voice English, SONAR embeddings:

| Model        | Text  | Speech |
|--------------|-------|--------|
| LiP (full-rank baseline) | 59.45 | 57.27 |
| FLiP rank-512 | **76.77** | **73.62** |
| FLiP rank-1024 | 77.29 | 74.09 |

Comparison with SpLiCE on span-aware accuracy (10k concept vocabulary):

| Method | Text  | Speech |
|--------|-------|--------|
| SpLiCE | 29.58 | 28.21  |
| FLiP   | **61.45** | **58.83** |

Cross-lingual and cross-modal analysis shows that SONAR embeddings have strong intra-language modality alignment but are English-biased across languages. See the paper for full results across SONAR, LaBSE, and Gemini embeddings.

---

## Installation

Requires Python >= 3.12 and PyTorch with CUDA 12.6.

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```
uv sync
```

---

## Data

Preprocessed SONAR embeddings and transcripts for Mozilla Common Voice v15 (EN) are available on HuggingFace: [BUT-FIT/FLiP-data](https://huggingface.co/datasets/BUT-FIT/FLiP-data).

---

## Usage

**1. Build vocabulary**

```
python scripts/build_cvect.py \
    --vocab_yaml configs/vocab_en.yaml \
    --data_yaml configs/datasets.yaml \
    --output_dir exp/cv_15/
```

**2. Train**

```
python lolm/train.py --config configs/train_lolm.yaml
```

Key config overrides:

```
python lolm/train.py --config configs/train_lolm.yaml \
    --name my_experiment \
    --rank 512 \
    --alpha 0.5 \
    --l1 1e-4
```

**3. Evaluate**

```
python scripts/evaluate_simple.py \
    --sdict exp/cv_15/en/rank_512_alpha_0.5_l1_1e-4_l2_0/checkpoints/model_state_dict_best_val_recall_avg.pt \
    --data_yaml configs/datasets.yaml \
    --dataset mcv_15_en_test \
    --text_id text_en \
    --topn 10
```

Pass `--entities_jsonl` to evaluate named-entity recall. Pass `--add_bias` to include the log-unigram prior in scoring. Pass `--save_details` to write per-document keyword results to JSON.

---

## Citation

```bibtex
@misc{kesiraju2026flip,
  title         = {{FLiP}: Towards understanding and interpreting multimodal multilingual sentence embeddings},
  author        = {Kesiraju, Santosh and Yusuf, Bolaji and Sedl{\'{a}}{\v{c}}ek, {\v{S}}imon and Plchot, Old{\v{r}}ich and Schwarz, Petr},
  year          = {2026},
  eprint        = {2026.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://github.com/BUTSpeechFIT/FLiP},
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

