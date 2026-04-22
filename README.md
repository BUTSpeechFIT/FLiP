
# FLiP: Factorized Linear Projection for Interpreting Multimodal Multilingual Sentence Embeddings

FLiP is a diagnostic tool for interpreting pretrained sentence embedding spaces. It trains a factorized log-linear model to recover lexical content (keywords) from sentence embeddings via a single linear projection — no fine-tuning of the encoder, no heuristics.

Under review:
> Santosh Kesiraju, Bolaji Yusuf, Simon Sedlacek, Oldrich Plchot, Petr Schwarz.
> *FLiP: Towards understanding and interpreting multimodal multilingual sentence embeddings.*
> Speech@FIT, Brno University of Technology.
> arXiv: [2604.18109](https://arxiv.org/abs/2604.18109)

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

## Trained checkpoints

| HF repo | Training data | Embedding | Rank | Size |
|---------|--------------|-----------|-----:|-----:|
| [BUT-FIT/FLiP-en-sonar](https://huggingface.co/BUT-FIT/FLiP-en-sonar) → `mcv15/rank-512/` | MCV v15 EN | SONAR | 512 | 207 MB |
| [BUT-FIT/FLiP-en-sonar](https://huggingface.co/BUT-FIT/FLiP-en-sonar) → `mcv15/rank-1024/` | MCV v15 EN | SONAR | 1024 | 414 MB |

Download with the `hf` CLI:

```bash
hf download BUT-FIT/FLiP-en-sonar mcv15/rank-512/model.safetensors \
                                   mcv15/rank-512/vocab.json \
                                   mcv15/rank-512/config.json
```

Or clone the full repo:

```bash
git clone https://huggingface.co/BUT-FIT/FLiP-en-sonar
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
python scripts/evaluate.py \
    --sdict mcv15/rank-512/model.safetensors \
    --vocab  mcv15/rank-512/vocab.json \
    --data_yaml configs/datasets.yaml \
    --dataset mcv_15_en_test \
    --text_id text_en \
    --topn 10 \
    --metrics all
```

- For real-world applications, play with `--topn N` -- you may also plot precision-recall curves as a function of `N`.

- To obtain accuracy pass `--metrics accuracy` -- here `--topn` does not matter because `n` is chosen based on in-vocab tokens per each sentence in the transcript.

Pass `--entities_jsonl` to evaluate named-entity recall. Pass `--add_bias` to include the log-unigram prior in scoring. Pass `--save_details` to write per-document keyword results to JSON.

---

## Citation

```bibtex
@misc{kesiraju2026flip,
  title         = {{FLiP}: Towards understanding and interpreting multimodal multilingual sentence embeddings},
  author        = {Kesiraju, Santosh and Yusuf, Bolaji and Sedl{\'{a}}{\v{c}}ek, {\v{S}}imon and Plchot, Old{\v{r}}ich and Schwarz, Petr},
  year          = {2026},
  eprint        = {2604.18109},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {http://arxiv.org/abs/2604.18109},
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

