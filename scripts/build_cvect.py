#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build count vectorizer using YAML config files.
"""

import os
import sys
import argparse
import json
import pickle
import logging

import yaml
from numpyencoder import NumpyEncoder
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from lolm.data.utils import load_text_and_strip, filter_data


def load_texts_from_yaml(data_yaml, vocab_config, logger):
    """Load texts from datasets.yaml based on vocab config."""

    # Load datasets
    with open(data_yaml, "r", encoding="utf-8") as f:
        datasets = yaml.safe_load(f)

    text_id = vocab_config["text_id"]
    sim_threshold = vocab_config["sim_threshold"]
    dataset_names = vocab_config["datasets"]

    logger.info(f"Loading texts with text_id={text_id}, sim_threshold>={sim_threshold}")
    logger.info(f"Datasets: {dataset_names}")

    all_texts = []

    for dataset_name in dataset_names:
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in {data_yaml}")

        dataset = datasets[dataset_name]
        logger.info(f"Processing dataset: {dataset_name}")

        # Find the text file
        text_file = None
        for text_entry in dataset["texts"]:
            if text_entry["id"] == text_id:
                text_file = text_entry["file"]
                break

        if text_file is None:
            raise ValueError(
                f"text_id '{text_id}' not found in dataset '{dataset_name}'"
            )

        # Find similarity score file if filtering is needed
        if sim_threshold > 0.0 and "sim_scores" in dataset:
            # Load similarity scores
            sim_score_file = None
            for sim_entry in dataset["sim_scores"]:
                # Use first similarity score file (could be made more specific)
                sim_score_file = sim_entry["file"]
                break

            if sim_score_file:
                logger.info(f"  Text: {text_file}")
                logger.info(f"  Sim scores: {sim_score_file}")
                texts = filter_data(
                    text_file,
                    sim_score_file,
                    sim_thresh=sim_threshold,
                    norm=None,
                    strip=True,
                )
                logger.info(f"  Loaded {len(texts)} texts (after sim filtering)")
            else:
                logger.warning("  No sim_scores found, loading all texts")
                texts = load_text_and_strip(text_file)
                logger.info(f"  Loaded {len(texts)} texts")
        else:
            texts = load_text_and_strip(text_file)
            logger.info(f"  Loaded {len(texts)} texts (no sim filtering)")

        all_texts.extend(texts)

    logger.info(f"Total texts loaded: {len(all_texts)}")
    return all_texts


def main():
    """main method"""

    args = parse_arguments()

    token_patterns = {"latin": r"(?u)\b\w\w+\b", "indic": r"[\S]+"}

    # Load vocab config from YAML if provided, otherwise build from arguments
    if args.vocab_yaml:
        with open(args.vocab_yaml, "r", encoding="utf-8") as f:
            vocab_config = yaml.safe_load(f)

        # Override config with command-line arguments
        if args.datasets:
            vocab_config["datasets"] = args.datasets
        if args.text_id:
            vocab_config["text_id"] = args.text_id
        if args.sim_threshold is not None:
            vocab_config["sim_threshold"] = args.sim_threshold
        if args.token_pattern:
            vocab_config["token_pattern"] = args.token_pattern
        if args.ngram_range:
            vocab_config["ngram_range"] = args.ngram_range
        if args.max_features is not None:
            vocab_config["max_features"] = args.max_features
        if args.min_df is not None:
            vocab_config["min_df"] = args.min_df
        if args.max_df is not None:
            vocab_config["max_df"] = args.max_df
        if args.output_dir:
            vocab_config["output_dir"] = args.output_dir
        if args.name:
            vocab_config["name"] = args.name
    else:
        # Build config entirely from command-line arguments
        vocab_config = {
            "name": args.name,
            "language": args.language or "unknown",
            "datasets": args.datasets,
            "text_id": args.text_id,
            "sim_threshold": args.sim_threshold,
            "token_pattern": args.token_pattern,
            "ngram_range": args.ngram_range,
            "lowercase": args.lowercase,
            "strip_accents": args.strip_accents,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "output_dir": args.output_dir,
        }

    # Setup output paths
    out_dir = os.path.join(vocab_config["output_dir"], vocab_config["name"])
    os.makedirs(out_dir, exist_ok=True)

    out_cvect = os.path.join(out_dir, "cvect.pkl")
    out_vocab_yaml = os.path.join(out_dir, "vocab_config.yaml")

    # Check if output directory already has files
    if os.path.exists(out_cvect) or os.path.exists(out_vocab_yaml):
        if not args.ovr:
            print(f"WARNING: Output directory {out_dir} already contains files.")
            print("Use --ovr to overwrite")
            sys.exit(0)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        filename=os.path.join(out_dir, "build_cvect.log"),
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info("scikit-learn version: %s", sklearn.__version__)
    logger.info(f"Vocab config: {args.vocab_yaml}")
    logger.info(f"Data config: {args.data_yaml}")
    logger.info(json.dumps(vocab_config, indent=2))

    # Load texts
    in_texts = load_texts_from_yaml(args.data_yaml, vocab_config, logger)

    if args.vocab_txt:
        logger.info(f"Loading vocabulary from {args.vocab_txt}")
        with open(args.vocab_txt, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}
        inferred_ngram_range = (1, 1)
        for word in vocab_list:
            n = len(word.split())
            if n > inferred_ngram_range[1]:
                inferred_ngram_range = (1, n)
        logger.info(f"Inferred ngram_range from vocab: {inferred_ngram_range}")
        logger.info(f"Vocabulary size from txt: {len(vocab_dict)}")
        cvect = CountVectorizer(
            ngram_range=tuple(vocab_config["ngram_range"]),
            lowercase=vocab_config.get("lowercase", True),
            strip_accents=vocab_config.get("strip_accents", None),
            token_pattern=token_patterns[vocab_config["token_pattern"]],
            vocabulary=vocab_dict,
        )
        dbyw = cvect.transform(in_texts)

    else:
        # Build CountVectorizer
        max_features = vocab_config.get("max_features", -1)
        cvect = CountVectorizer(
            ngram_range=tuple(vocab_config["ngram_range"]),
            max_features=None if max_features == -1 else max_features,
            min_df=vocab_config.get("min_df", 1),
            max_df=vocab_config.get("max_df", 1.0),
            lowercase=vocab_config.get("lowercase", True),
            strip_accents=vocab_config.get("strip_accents", None),
            token_pattern=token_patterns[vocab_config["token_pattern"]],
        )

    # Fit vectorizer
    dbyw = cvect.fit_transform(in_texts)
    logger.info(f"Number of documents: {dbyw.shape[0]:,}")
    vsize = len(cvect.vocabulary_)
    logger.info(f"Vocab size: {vsize:,}")
    logger.info(f"Number of tokens: {dbyw.sum():,}")

    # Save outputs
    with open(out_cvect, "wb") as fpw:
        pickle.dump(cvect, fpw)
    logger.info("Saved cvect in %s", out_cvect)

    with open(
        os.path.join(out_dir, f"vocab_{vsize}.json"), "w", encoding="utf-8"
    ) as fpw:
        json.dump(
            cvect.vocabulary_, fpw, indent=2, ensure_ascii=False, cls=NumpyEncoder
        )
    logger.info(f"Saved vocab_{vsize}.json")

    # Save the vocabulary config used
    with open(out_vocab_yaml, "w", encoding="utf-8") as fpw:
        yaml.dump(vocab_config, fpw, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved vocab config in {out_vocab_yaml}")

    logger.info("Vocabulary building complete")


def parse_arguments():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vocab_yaml",
        help="Path to vocabulary config YAML (optional if all required args provided)",
    )
    parser.add_argument(
        "--data_yaml",
        required=True,
        help="Path to datasets YAML",
    )
    parser.add_argument(
        "--ovr",
        action="store_true",
        help="Overwrite existing output files",
    )

    custom_group = parser.add_argument_group(
        "Custom Vocabulary Config",
        "Specify vocabulary configuration parameters directly (required if --vocab_yaml not provided)",
    )

    # Vocabulary configuration parameters (required if --vocab_yaml not provided)
    custom_group.add_argument(
        "--name",
        help="Vocabulary name (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--language",
        help="Language code (e.g., en, cs)",
    )
    custom_group.add_argument(
        "--datasets",
        nargs="+",
        help="List of dataset names (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--text_id",
        help="Text ID to use (e.g., text_en) (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--sim_threshold",
        type=float,
        help="Similarity threshold (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--token_pattern",
        choices=["latin", "indic"],
        help="Tokenization pattern (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--ngram_range",
        type=int,
        nargs=2,
        help="N-gram range [min, max] (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--lowercase",
        type=bool,
        default=True,
        help="Convert to lowercase",
    )
    custom_group.add_argument(
        "--strip_accents",
        choices=["ascii", "unicode"],
        help="Strip accents method",
    )
    custom_group.add_argument(
        "--max_features",
        type=int,
        help="Max vocabulary size (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--min_df",
        type=int,
        help="Min document frequency (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--max_df",
        type=float,
        help="Max document frequency (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--output_dir",
        help="Output directory (required if no --vocab_yaml)",
    )
    custom_group.add_argument(
        "--vocab_txt", help="Path to load vocab from txt file (optional)"
    )

    args = parser.parse_args()

    # Validate: either vocab_yaml or all required args must be provided
    if not args.vocab_yaml:
        required_args = [
            "name",
            "datasets",
            "text_id",
            "sim_threshold",
            "token_pattern",
            "ngram_range",
            "max_features",
            "min_df",
            "max_df",
            "output_dir",
        ]
        missing = [arg for arg in required_args if getattr(args, arg) is None]
        if missing:
            parser.error(
                f"Without --vocab_yaml, these arguments are required: {', '.join(missing)}"
            )

    return args


if __name__ == "__main__":
    main()
