import os
import json
import logging
import pickle
from typing import List, Union
import yaml

# import numpy as np
from numpyencoder import NumpyEncoder
from lolm.data.datasets import EmbBoWDataset
from lolm.data.utils import load_json_to_list, load_yaml_to_list
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def build_embow_dataset(input_json, args, cvect):
    """Build EmbBoW Dataset"""

    emb1_files, emb2_files, text_files, sim_files, langs = load_json_to_list(
        input_json, args.emb_ixs, args.target_text_idx, args.tmp_dir
    )

    # vocab = None
    # if args.vocab_json:
    #    with open(args.vocab_json, 'r', encoding='utf-8') as fpr:
    #        vocab = json.load(fpr)

    dset = EmbBoWDataset(
        text_files,
        emb1_files,
        emb2_files,
        sim_files,
        cvect,
        sim_thresh=args.sim_thresh,
        # min_df=args.min_df,
        # max_df=args.max_df,
        # max_vocab=args.max_vocab,
        # lowercase=(not args.true_case),
        # ngram_range=args.ngram_range,
        # stop_words=args.stop_words,
        # vocab=vocab,
        norm=args.norm,
        langs=langs,
    )

    if "exp_dir" in args.__dict__:
        with open(
            os.path.join(args.exp_dir, "vocab.json"), "w", encoding="utf-8"
        ) as fpw:
            json.dump(
                dset.cvect.vocabulary_,
                fpw,
                indent=2,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )

    return dset


def build_embow_dataset_from_yaml(
    data_yaml: str,
    vocab_yaml_or_cvect: Union[str, CountVectorizer],
    dataset_names: List[str],
    emb_pair: List[str],
    target_text_id: str,
    sim_thresh: float = 0.7,
    norm: Union[None, str] = None,
    tmp_dir: str = None,
    exp_dir: str = None,
) -> EmbBoWDataset:
    """Build EmbBoW Dataset from YAML configurations.

    Args:
        data_yaml: Path to datasets.yaml file
        vocab_yaml_or_cvect: Path to vocab_config.yaml or CountVectorizer object
        dataset_names: List of dataset names to load
        emb_pair: List of two embedding IDs (e.g., ['text_en', 'speech_en'])
        target_text_id: Text ID to use as target (e.g., 'text_en')
        sim_thresh: Similarity threshold for filtering
        norm: Normalization method for embeddings
        tmp_dir: Optional temporary directory
        exp_dir: Optional experiment directory to save vocab.json

    Returns:
        EmbBoWDataset instance
    """
    logger.info("Building EmbBoW dataset from YAML configs")

    # Load CountVectorizer
    if isinstance(vocab_yaml_or_cvect, str):
        # Load from vocab config YAML - find the cvect.pkl file
        with open(vocab_yaml_or_cvect, "r", encoding="utf-8") as f:
            vocab_config = yaml.safe_load(f)

        # cvect.pkl should be in the same directory as vocab_config.yaml
        vocab_dir = os.path.dirname(vocab_yaml_or_cvect)
        cvect_path = os.path.join(vocab_dir, "cvect.pkl")

        if not os.path.exists(cvect_path):
            raise FileNotFoundError(
                f"CountVectorizer not found at {cvect_path}. "
                f"Expected it to be in the same directory as {vocab_yaml_or_cvect}"
            )

        logger.info("Loading CountVectorizer from %s", cvect_path)
        with open(cvect_path, "rb") as f:
            cvect = pickle.load(f)
    else:
        cvect = vocab_yaml_or_cvect

    # Load data files from YAML
    emb1_files, emb2_files, text_files, sim_files, langs = load_yaml_to_list(
        data_yaml, dataset_names, emb_pair, target_text_id, tmp_dir
    )

    # Create dataset
    dset = EmbBoWDataset(
        text_files,
        emb1_files,
        emb2_files,
        sim_files,
        cvect,
        sim_thresh=sim_thresh,
        norm=norm,
        langs=langs,
    )

    # Save vocabulary if exp_dir provided
    if exp_dir:
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "vocab.json"), "w", encoding="utf-8") as fpw:
            json.dump(
                dset.cvect.vocabulary_,
                fpw,
                indent=2,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )
        logger.info("Saved vocabulary to %s", os.path.join(exp_dir, "vocab.json"))

    return dset
