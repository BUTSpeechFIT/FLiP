#!/usr/bin/env python3
"""Evaluate LoLM/FactLoLM keyword extraction using precision, recall, and entity metrics."""

import os
import json
import argparse
from time import time

import numpy as np
import torch
from tqdm import tqdm
import yaml

from lolm.data.utils import get_int2vocab, load_data_and_embs, remove_punc
from lolm.utils import create_logger


def load_entities_from_jsonl(jsonl_file):
    """Load named entities from a JSONL file.

    Args:
        jsonl_file: Path to JSONL file with entity data.

    Returns:
        Tuple of (text_to_entities, line_to_entities) dicts.
    """
    text_to_entities = {}
    line_to_entities = {}

    if not os.path.exists(jsonl_file):
        return text_to_entities, line_to_entities

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            entity_names = data.get("entity_names", [])
            if not entity_names:
                continue
            text_key = remove_punc(data.get("text", "").strip()).lower()
            line_num = data.get("line_number")
            text_to_entities[text_key] = entity_names
            if line_num is not None:
                line_to_entities[line_num] = entity_names

    return text_to_entities, line_to_entities


def load_input_from_yaml(data_yaml, dataset_names=None):
    """Load dataset file paths from a YAML config.

    Args:
        data_yaml: Path to datasets YAML file.
        dataset_names: Optional list of dataset names to select.

    Returns:
        Dict keyed by dataset name with file paths and metadata.
    """
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if dataset_names:
        missing = [n for n in dataset_names if n not in data]
        if missing:
            raise ValueError(f"Datasets not found in {data_yaml}: {missing}")
        selected = {n: data[n] for n in dataset_names}
    else:
        selected = data

    input_data = {}
    for name, dataset in selected.items():
        embeddings = dataset.get("embeddings", [])
        texts = dataset.get("texts", [])
        sim_scores = dataset.get("sim_scores", [])
        input_data[name] = {
            "embs": [e.get("file") for e in embeddings],
            "emb_ids": [e.get("id") for e in embeddings],
            "mods": [e.get("modality") for e in embeddings],
            "langs": [e.get("language") for e in embeddings],
            "transcripts": [t.get("file") for t in texts],
            "text_ids": [t.get("id") for t in texts],
            "text_langs": [t.get("language") for t in texts],
            "sim_scores": [s.get("file") for s in sim_scores],
            "sim_pairs": [s.get("pair") for s in sim_scores],
        }

    return input_data


def compute_standard_metrics(doc, keywords):
    """Compute token-level precision, recall, and F1 against a reference document.

    Args:
        doc: Reference document string.
        keywords: List of extracted keyword strings.

    Returns:
        Dict with precision, recall, f1, n_ref, n_hyp, n_overlap.
    """
    ref_toks = set(doc.lower().split())
    hyp_toks = set(kw.lower() for kw in keywords)
    overlap = ref_toks & hyp_toks

    precision = len(overlap) / len(hyp_toks) if hyp_toks else 0.0
    recall = len(overlap) / len(ref_toks) if ref_toks else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_ref": len(ref_toks),
        "n_hyp": len(hyp_toks),
        "n_overlap": len(overlap),
    }


def compute_entity_metrics(entity_names, keywords):
    """Compute entity-based precision and recall.

    Strict recall: fraction of entities where all tokens appear in keywords.
    Partial recall: average token coverage per entity.
    Precision: fraction of keywords that matched at least one entity token.

    Args:
        entity_names: List of ground-truth entity name strings.
        keywords: List of extracted keyword strings.

    Returns:
        Dict with entity precision, strict/partial recall, strict/partial F1.
    """
    hyp_keywords = set(kw.lower() for kw in keywords)
    strict_matched = set()
    matched_keywords = set()
    partial_scores = []

    for entity in entity_names:
        entity_lower = entity.lower()
        entity_tokens = entity_lower.split()
        matched_count = 0

        for token in entity_tokens:
            if token in hyp_keywords:
                matched_count += 1
                matched_keywords.add(token)

        if matched_count == len(entity_tokens):
            strict_matched.add(entity_lower)

        partial_scores.append(
            matched_count / len(entity_tokens) if entity_tokens else 0.0
        )

    n_entities = len(entity_names)
    strict_recall = len(strict_matched) / n_entities if n_entities > 0 else 0.0
    partial_recall = (
        sum(partial_scores) / len(partial_scores) if partial_scores else 0.0
    )
    precision = len(matched_keywords) / len(hyp_keywords) if hyp_keywords else 0.0

    strict_f1 = (
        2 * precision * strict_recall / (precision + strict_recall)
        if (precision + strict_recall) > 0
        else 0.0
    )
    partial_f1 = (
        2 * precision * partial_recall / (precision + partial_recall)
        if (precision + partial_recall) > 0
        else 0.0
    )

    return {
        "entity_precision": precision,
        "entity_recall_strict": strict_recall,
        "entity_recall_partial": partial_recall,
        "entity_f1_strict": strict_f1,
        "entity_f1_partial": partial_f1,
        "n_entities": n_entities,
        "n_keywords": len(hyp_keywords),
        "n_strict_matched": len(strict_matched),
        "n_matched_keywords": len(matched_keywords),
    }


def extract_topn_keywords(E, embs, topn, batch_size, add_bias=False, b=None):
    """Extract top-N keyword indices for each document.

    Args:
        E: Projection matrix (vocab_size x emb_dim) on device.
        embs: Document embeddings as numpy array (n_docs x emb_dim).
        topn: Number of keywords to extract per document.
        batch_size: Number of documents processed per batch.
        add_bias: Whether to add the bias vector to scores.
        b: Bias vector (vocab_size x 1) on device; required if add_bias=True.

    Returns:
        List of LongTensors of length topn, one per document.
    """
    device = E.device
    embs_torch = torch.from_numpy(embs).to(device=device, dtype=E.dtype)
    all_kw_ixs = []

    for start in tqdm(range(0, len(embs), batch_size), desc="Extracting keywords"):
        batch = embs_torch[start : start + batch_size]
        scores = E @ batch.T
        if add_bias and b is not None:
            scores = scores + b
        _, top_ixs = torch.topk(scores, k=topn, dim=0, sorted=True)
        for i in range(batch.shape[0]):
            all_kw_ixs.append(top_ixs[:, i])

    return all_kw_ixs


def evaluate_all(docs, all_kw_ixs, int2vocab, topn, logger, entity_names_list=None):
    """Evaluate all documents and aggregate metrics.

    Args:
        docs: List of document strings.
        all_kw_ixs: List of LongTensors of keyword indices, one per document.
        int2vocab: Dict mapping integer index to vocabulary word.
        topn: Number of keywords to use per document.
        logger: Logger instance.
        entity_names_list: Optional list of entity name lists, one per document.

    Returns:
        Tuple of (results list, summary dict).
    """
    has_entities = entity_names_list is not None
    eval_mode = "entity" if has_entities else "standard"

    results = []
    metric_lists = {"precision": [], "recall": [], "f1": []}
    entity_metric_keys = [
        "entity_precision",
        "entity_recall_strict",
        "entity_recall_partial",
        "entity_f1_strict",
        "entity_f1_partial",
    ]
    entity_metric_lists = {k: [] for k in entity_metric_keys}

    for i in tqdm(range(len(docs)), desc=f"Evaluating (topn={topn}, mode={eval_mode})"):
        keywords = [int2vocab[ix.item()] for ix in all_kw_ixs[i]]
        metrics = compute_standard_metrics(docs[i], keywords)

        entity_names = entity_names_list[i] if has_entities else None
        if entity_names is not None:
            metrics.update(compute_entity_metrics(entity_names, keywords))

        result = {"doc": docs[i], "keywords": keywords, "metrics": metrics}
        if entity_names is not None:
            result["entity_names"] = entity_names
        results.append(result)

        for key in metric_lists:
            metric_lists[key].append(metrics[key])
        if has_entities:
            for key in entity_metric_keys:
                entity_metric_lists[key].append(metrics[key])

    def aggregate(values):
        arr = np.array(values, dtype=np.float32)
        n = len(arr)
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "se": round(float(arr.std() / np.sqrt(n)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }

    summary = {
        "evaluation_mode": eval_mode,
        "n_documents": len(docs),
        "topn": topn,
    }
    for key in metric_lists:
        summary[key] = aggregate(metric_lists[key])
    if has_entities:
        for key in entity_metric_keys:
            summary[key] = aggregate(entity_metric_lists[key])

    logger.info(json.dumps(summary))
    _print_summary_table(summary, has_entities)

    return results, summary


def _print_summary_table(summary, has_entities):
    """Print evaluation metrics in a compact table.

    Args:
        summary: Summary dict from evaluate_all.
        has_entities: Whether entity metrics are present in summary.
    """
    print("\n" + "=" * 72)
    print(f"EVALUATION MODE: {summary['evaluation_mode'].upper()}")
    print("=" * 72)
    print(f"{'Metric':<26} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 72)

    standard_metrics = [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    for label, key in standard_metrics:
        m = summary[key]
        print(
            f"{label:<26} {m['mean']:>8.4f} {m['std']:>8.4f} {m['min']:>8.4f} {m['max']:>8.4f}"
        )

    if has_entities:
        print("-" * 72)
        entity_metrics = [
            ("Entity Precision", "entity_precision"),
            ("Entity Recall (strict)", "entity_recall_strict"),
            ("Entity Recall (partial)", "entity_recall_partial"),
            ("Entity F1 (strict)", "entity_f1_strict"),
            ("Entity F1 (partial)", "entity_f1_partial"),
        ]
        for label, key in entity_metrics:
            m = summary[key]
            print(
                f"{label:<26} {m['mean']:>8.4f} {m['std']:>8.4f} {m['min']:>8.4f} {m['max']:>8.4f}"
            )

    print("=" * 72)
    print(f"Documents: {summary['n_documents']}  |  Top-N: {summary['topn']}")
    print("=" * 72 + "\n")


def _save_json(path, data):
    """Save data to a JSON file, converting numpy/tensor values automatically.

    Args:
        path: Output file path.
        data: Data to serialize.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
        )


def run_evaluation(
    docs, embs, text_f, emb_f, basename, args, out_dir, int2vocab, E, b, logger
):
    """Run keyword extraction and evaluation for one embedding file.

    Args:
        docs: List of document strings.
        embs: Numpy array of embeddings (n_docs x emb_dim).
        text_f: Path to the transcript text file.
        emb_f: Path to the embedding file (used for line-number matching).
        basename: Base name for output files.
        args: Parsed argument namespace.
        out_dir: Output directory.
        int2vocab: Dict mapping index to vocabulary word.
        E: Projection matrix on device.
        b: Bias vector on device, or None.
        logger: Logger instance.
    """
    # Build line-number index for entity matching
    with open(text_f, "r", encoding="utf-8") as f:
        original_lines = [remove_punc(line.strip()) for line in f]

    doc_line_numbers = []
    for doc in docs:
        matched = False
        for line_num, orig_line in enumerate(original_lines, 1):
            if orig_line is not None and orig_line == doc.strip():
                doc_line_numbers.append(line_num)
                original_lines[line_num - 1] = None
                matched = True
                break
        if not matched:
            doc_line_numbers.append(None)

    # Optional sentence-length filtering
    if args.msl > 1 or args.xsl < 1000:
        logger.info(
            "Filtering by sentence length: %d <= tokens <= %d", args.msl, args.xsl
        )
        filtered = [
            (doc, embs[i], doc_line_numbers[i])
            for i, doc in enumerate(docs)
            if args.msl <= len(doc.split()) <= args.xsl
        ]
        if filtered:
            docs, embs_list, doc_line_numbers = zip(*filtered)
            docs = list(docs)
            embs = np.array(embs_list)
            doc_line_numbers = list(doc_line_numbers)
        else:
            docs, embs, doc_line_numbers = [], np.array([]), []
        logger.info("After filtering: %d documents", len(docs))

    # Load named entities
    text_to_entities = {}
    line_to_entities = {}
    if args.entities_jsonl:
        logger.info("Loading entities from %s", args.entities_jsonl)
        text_to_entities, line_to_entities = load_entities_from_jsonl(
            args.entities_jsonl
        )
        logger.info("Loaded entities for %d documents", len(text_to_entities))

    # Extract top-N keywords
    stime = time()
    logger.info("Extracting top-%d keywords", args.topn)
    all_kw_ixs = extract_topn_keywords(
        E, embs, args.topn, args.batch_size, args.add_bias, b
    )
    logger.info("Extraction completed in %.2fs", time() - stime)

    pr_dir = os.path.join(out_dir, "precision_recall")

    # Standard evaluation
    logger.info("Running standard evaluation")
    results, summary = evaluate_all(docs, all_kw_ixs, int2vocab, args.topn, logger)

    if args.save_details:
        _save_json(
            os.path.join(pr_dir, f"{basename}_topn{args.topn}_results.json"),
            results,
        )
    _save_json(
        os.path.join(pr_dir, f"{basename}_topn{args.topn}_summary.json"),
        {"config": vars(args), "metrics": summary},
    )

    # Entity evaluation (subset of docs that have entities)
    if text_to_entities or line_to_entities:
        entity_docs = []
        entity_kw_ixs = []
        entity_names_list = []

        for i, doc in enumerate(docs):
            doc_key = remove_punc(doc.strip()).lower()
            if doc_key in text_to_entities:
                entity_docs.append(doc)
                entity_kw_ixs.append(all_kw_ixs[i])
                entity_names_list.append(text_to_entities[doc_key])
            elif (
                doc_line_numbers[i] is not None
                and doc_line_numbers[i] in line_to_entities
            ):
                entity_docs.append(doc)
                entity_kw_ixs.append(all_kw_ixs[i])
                entity_names_list.append(line_to_entities[doc_line_numbers[i]])

        logger.info(
            "Entity evaluation: %d / %d docs have entities",
            len(entity_docs),
            len(docs),
        )

        if not entity_docs:
            logger.warning("No documents with entities found - check text matching")
            return

        entity_results, entity_summary = evaluate_all(
            entity_docs,
            entity_kw_ixs,
            int2vocab,
            args.topn,
            logger,
            entity_names_list=entity_names_list,
        )

        if args.save_details:
            _save_json(
                os.path.join(pr_dir, f"{basename}_topn{args.topn}_entity_results.json"),
                entity_results,
            )
        _save_json(
            os.path.join(pr_dir, f"{basename}_topn{args.topn}_entity_summary.json"),
            {
                "config": vars(args),
                "metrics": entity_summary,
                "n_entity_docs": len(entity_docs),
            },
        )


def main():
    """Main entry point."""
    args = parse_arguments()
    stime = time()

    out_dir = os.path.join(os.path.dirname(args.sdict), "../results/")
    os.makedirs(out_dir, exist_ok=True)

    logger = create_logger(
        os.path.join(
            os.path.dirname(args.sdict),
            f"../logs/eval_{args.text_id}_topn{args.topn}",
        ),
        args.verbose,
    )
    logger.info(json.dumps(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cpu":
        logger.warning("GPU not available, falling back to CPU")

    # Load model checkpoint
    logger.info("Loading model checkpoint from %s", args.sdict)
    ckpt = torch.load(args.sdict, map_location=device, weights_only=True)

    # Extract projection matrix E (supports both compiled and non-compiled checkpoints)
    if "E" in ckpt:
        E = ckpt["E"].float()
    elif "_orig_mod.E" in ckpt:
        E = ckpt["_orig_mod.E"].float()
    elif "E1" in ckpt:
        E = (ckpt["E1"] @ ckpt["E2"]).float()
    elif "_orig_mod.E1" in ckpt:
        E = (ckpt["_orig_mod.E1"] @ ckpt["_orig_mod.E2"]).float()
    else:
        raise ValueError("Could not find E or E1/E2 in checkpoint")

    # Extract bias vector
    b = None
    if args.add_bias:
        if "b" in ckpt:
            b = ckpt["b"].float()
        elif "_orig_mod.b" in ckpt:
            b = ckpt["_orig_mod.b"].float()
        else:
            raise ValueError("Bias not found in checkpoint but --add_bias was set")

    cvect_path = os.path.join(os.path.dirname(args.sdict), "../cvect.pkl")
    int2vocab = get_int2vocab(cvect_path)
    logger.info("Vocabulary size: %d", len(int2vocab))

    input_data = load_input_from_yaml(args.data_yaml, [args.dataset])

    for dataset_name, ds in input_data.items():
        logger.info("Processing dataset: %s", dataset_name)

        emb_by_id = dict(zip(ds["emb_ids"], ds["embs"]))
        mod_by_id = dict(zip(ds["emb_ids"], ds["mods"]))
        lang_by_emb_id = dict(zip(ds["emb_ids"], ds["langs"]))
        text_by_id = dict(zip(ds["text_ids"], ds["transcripts"]))
        lang_by_text_id = dict(zip(ds["text_ids"], ds["text_langs"]))

        if args.text_id not in text_by_id:
            raise ValueError(
                f"Text ID '{args.text_id}' not found in dataset '{dataset_name}'"
            )

        text_f = text_by_id[args.text_id]

        if not ds["sim_scores"]:
            raise ValueError(f"No sim_scores found for dataset '{dataset_name}'")

        for sim_score_f, sim_pair in zip(ds["sim_scores"], ds["sim_pairs"]):
            emb_id_1, emb_id_2 = sim_pair
            logger.info(
                "Embedding pair: %s (%s) <-> %s (%s)",
                lang_by_emb_id[emb_id_1],
                mod_by_id[emb_id_1],
                lang_by_emb_id[emb_id_2],
                mod_by_id[emb_id_2],
            )
            logger.info("Target text: %s", lang_by_text_id[args.text_id])

            docs, embs1, embs2, _ = load_data_and_embs(
                text_f,
                emb_by_id[emb_id_1],
                emb_by_id[emb_id_2],
                sim_score_f,
                norm=args.norm,
                strip=True,
                sim_thresh=0.0,
            )
            logger.info("Loaded %d documents", len(docs))

            for emb_id, emb_f, embs in (
                (emb_id_1, emb_by_id[emb_id_1], embs1),
                (emb_id_2, emb_by_id[emb_id_2], embs2),
            ):
                if args.out_base:
                    basename = args.out_base
                    os.makedirs(
                        os.path.join(out_dir, "precision_recall"), exist_ok=True
                    )
                else:
                    emb_base = os.path.basename(emb_f).replace(".npy", "")
                    basename = f"{dataset_name}/{emb_base}"
                    os.makedirs(
                        os.path.join(out_dir, "precision_recall", dataset_name),
                        exist_ok=True,
                    )

                run_evaluation(
                    docs,
                    embs,
                    text_f,
                    emb_f,
                    basename,
                    args,
                    out_dir,
                    int2vocab,
                    E,
                    b,
                    logger,
                )

    total_time = time() - stime
    logger.info("Total time: %.2fs", total_time)
    print(f"Results saved to {out_dir}")
    print(f"Total time: {total_time:.2f}s")


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--sdict",
        required=True,
        help="Path to model checkpoint state_dict (.pt)",
    )
    parser.add_argument(
        "--data_yaml",
        required=True,
        help="Path to datasets YAML file",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name to evaluate (e.g. mcv_15_en_test)",
    )
    parser.add_argument(
        "--text_id",
        required=True,
        help="Text ID for the target transcript (e.g. text_en)",
    )

    # Entity evaluation
    parser.add_argument(
        "--entities_jsonl",
        type=str,
        default=None,
        help="JSONL file with named entities (optional)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="Number of keywords to extract per document",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6000,
        help="Batch size for keyword extraction",
    )

    # Sentence-length filtering
    parser.add_argument(
        "--msl",
        type=int,
        default=1,
        help="Minimum sentence length in tokens",
    )
    parser.add_argument(
        "--xsl",
        type=int,
        default=1000,
        help="Maximum sentence length in tokens",
    )

    # Model options
    parser.add_argument(
        "--add_bias",
        action="store_true",
        help="Add bias vector to logits",
    )
    parser.add_argument(
        "--norm",
        default=None,
        choices=[None, "l2"],
        help="Sentence embedding normalization",
    )

    # Output options
    parser.add_argument(
        "--out_base",
        type=str,
        default="",
        help="Basename for output files (auto-generated if not set)",
    )
    parser.add_argument(
        "--save_details",
        action="store_true",
        help="Save per-document keyword results to JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    return args


if __name__ == "__main__":
    main()
