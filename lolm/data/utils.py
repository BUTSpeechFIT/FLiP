import os
import json
import pickle

# import shutil
import logging
import string
import re
from typing import Union, List, Tuple
import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def load_json_to_list(
    input_json, emb_ixs: List[int], target_text_idx: int, tmp_dir=None
):
    """Load file paths from json to lists"""

    logger.info("Loading json file: %s", input_json)
    logger.info("Using emb ixs: %s", json.dumps(emb_ixs))
    logger.info("Using target_text_idx: %d", target_text_idx)

    text_files = []
    emb1_files = []
    emb2_files = []
    sim_files = []
    langs = []
    keys = []
    data_json = {}
    with open(input_json, "r", encoding="utf-8") as fpr:
        data_json = json.load(fpr)

    lang = ""
    for key in data_json:
        assert (
            len(data_json[key]["embs"]) >= 2
        ), f"{key} must contain list of atleast 2 embs npy file paths"

        logger.info(
            "Language for target_text_idx %d is %s",
            target_text_idx,
            data_json[key]["langs"][target_text_idx],
        )

        if lang == "":
            lang = data_json[key]["langs"][target_text_idx]

        assert (
            lang == data_json[key]["langs"][target_text_idx]
        ), f"Target text idx language mismatch for key {key}: {lang} != {data_json[key]['langs'][target_text_idx]}"

        keys.append(key)
        emb1_files.append(data_json[key]["embs"][emb_ixs[0]])
        emb2_files.append(data_json[key]["embs"][emb_ixs[1]])
        text_files.append(data_json[key]["transcripts"][target_text_idx])
        langs.append(data_json[key]["langs"][emb_ixs[0]])
        langs.append(data_json[key]["langs"][emb_ixs[1]])

        assert (
            len(data_json[key]["sim_scores"]) == 1
        ), f"Expected only 1 sim score file in {input_json}, key: {key}"
        sim_files.append(data_json[key]["sim_scores"][0])

        logger.info("Loading key: %s", key)
        logger.info("Embeddings1: %s", emb1_files[-1])
        logger.info("Embeddings2: %s", emb2_files[-1])
        logger.info("Target text: %s", text_files[-1])

    if tmp_dir:
        logger.info("Copying files to tmp dir: %s", tmp_dir)
        tmp_emb1_files = []
        tmp_emb2_files = []
        tmp_text_files = []
        tmp_sim_files = []
        for i, _ in enumerate(emb1_files):
            tmp_emb1_files.append(
                os.path.join(tmp_dir, f"{keys[i]}_" + os.path.basename(emb1_files[i]))
            )
            tmp_emb2_files.append(
                os.path.join(tmp_dir, f"{keys[i]}_" + os.path.basename(emb2_files[i]))
            )
            tmp_text_files.append(
                os.path.join(tmp_dir, f"{keys[i]}_" + os.path.basename(text_files[i]))
            )
            tmp_sim_files.append(
                os.path.join(tmp_dir, f"{keys[i]}_" + os.path.basename(sim_files[i]))
            )

            # use shutil instead of os.system for copying files
            # shutil.copy(emb1_files[i], tmp_emb1_files[i])
            # shutil.copy(emb2_files[i], tmp_emb2_files[i])
            # shutil.copy(text_files[i], tmp_text_files[i])
            # shutil.copy(sim_files[i], tmp_sim_files[i])

            os.system(f"cp -uv {emb1_files[i]} {tmp_emb1_files[-1]}")
            os.system(f"cp -uv {emb2_files[i]} {tmp_emb2_files[-1]}")
            os.system(f"cp -uv {text_files[i]} {tmp_text_files[-1]}")
            os.system(f"cp -uv {sim_files[i]} {tmp_sim_files[-1]}")

        logger.info("Copied files to tmp dir %s", tmp_dir)

        return tmp_emb1_files, tmp_emb2_files, tmp_text_files, tmp_sim_files, langs

    else:
        return emb1_files, emb2_files, text_files, sim_files, langs


def load_yaml_to_list(
    data_yaml: str,
    dataset_names: List[str],
    emb_pair: List[str],
    target_text_id: str,
    tmp_dir: str = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Load file paths from YAML datasets config using named IDs.

    Args:
        data_yaml: Path to datasets.yaml file
        dataset_names: List of dataset names to load (e.g., ['mozilla_cv15_en'])
        emb_pair: List of two embedding IDs (e.g., ['text_en', 'speech_en'])
        target_text_id: Text ID to use as target (e.g., 'text_en')
        tmp_dir: Optional temporary directory to copy files to

    Returns:
        Tuple of (emb1_files, emb2_files, text_files, sim_files, langs)
    """
    logger.info("Loading datasets from YAML: %s", data_yaml)
    logger.info("Dataset names: %s", json.dumps(dataset_names))
    logger.info("Embedding pair: %s", json.dumps(emb_pair))
    logger.info("Target text ID: %s", target_text_id)

    assert len(emb_pair) == 2, "emb_pair must contain exactly 2 embedding IDs"

    # Load datasets YAML
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    text_files = []
    emb1_files = []
    emb2_files = []
    sim_files = []
    langs = []

    # Process each requested dataset
    for dataset_name in dataset_names:
        if dataset_name not in data:
            raise ValueError(f"Dataset '{dataset_name}' not found in {data_yaml}")

        dataset = data[dataset_name]
        logger.info("Loading dataset: %s", dataset_name)

        # Find embeddings by ID
        emb1_entry = None
        emb2_entry = None
        for emb in dataset["embeddings"]:
            if emb["id"] == emb_pair[0]:
                emb1_entry = emb
            if emb["id"] == emb_pair[1]:
                emb2_entry = emb

        if not emb1_entry:
            raise ValueError(
                f"Embedding '{emb_pair[0]}' not found in dataset '{dataset_name}'"
            )
        if not emb2_entry:
            raise ValueError(
                f"Embedding '{emb_pair[1]}' not found in dataset '{dataset_name}'"
            )

        # Find target text
        text_entry = None
        for text in dataset["texts"]:
            if text["id"] == target_text_id:
                text_entry = text
                break

        if not text_entry:
            raise ValueError(
                f"Text '{target_text_id}' not found in dataset '{dataset_name}'"
            )

        # Find similarity scores for this embedding pair
        sim_entry = None
        if "sim_scores" in dataset:
            for sim in dataset["sim_scores"]:
                # Check if this sim_score matches our embedding pair (order doesn't matter)
                if set(sim["pair"]) == set(emb_pair):
                    sim_entry = sim
                    break

        if not sim_entry:
            raise ValueError(
                f"Similarity scores for pair {emb_pair} not found in dataset '{dataset_name}'"
            )

        # Add to lists
        emb1_files.append(emb1_entry["file"])
        emb2_files.append(emb2_entry["file"])
        text_files.append(text_entry["file"])
        sim_files.append(sim_entry["file"])
        langs.append(emb1_entry["language"])
        langs.append(emb2_entry["language"])

        logger.info("  Embedding 1 (%s): %s", emb_pair[0], emb1_files[-1])
        logger.info("  Embedding 2 (%s): %s", emb_pair[1], emb2_files[-1])
        logger.info("  Target text (%s): %s", target_text_id, text_files[-1])
        logger.info("  Similarity scores: %s", sim_files[-1])

    # Copy to tmp dir if requested
    if tmp_dir:
        logger.info("Copying files to tmp dir: %s", tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_emb1_files = []
        tmp_emb2_files = []
        tmp_text_files = []
        tmp_sim_files = []

        for i, dataset_name in enumerate(dataset_names):
            tmp_emb1_files.append(
                os.path.join(
                    tmp_dir, f"{dataset_name}_" + os.path.basename(emb1_files[i])
                )
            )
            tmp_emb2_files.append(
                os.path.join(
                    tmp_dir, f"{dataset_name}_" + os.path.basename(emb2_files[i])
                )
            )
            tmp_text_files.append(
                os.path.join(
                    tmp_dir, f"{dataset_name}_" + os.path.basename(text_files[i])
                )
            )
            tmp_sim_files.append(
                os.path.join(
                    tmp_dir, f"{dataset_name}_" + os.path.basename(sim_files[i])
                )
            )

            os.system(f"cp -uv {emb1_files[i]} {tmp_emb1_files[-1]}")
            os.system(f"cp -uv {emb2_files[i]} {tmp_emb2_files[-1]}")
            os.system(f"cp -uv {text_files[i]} {tmp_text_files[-1]}")
            os.system(f"cp -uv {sim_files[i]} {tmp_sim_files[-1]}")

        logger.info("Copied files to tmp dir %s", tmp_dir)
        return tmp_emb1_files, tmp_emb2_files, tmp_text_files, tmp_sim_files, langs
    else:
        return emb1_files, emb2_files, text_files, sim_files, langs


def remove_punc(line):
    line = line.translate(
        str.maketrans(
            "",
            "",
            string.punctuation + "’“”‘‘[]“" + "\u200c" + "\u200b" + "\u2060" + "—",
        )
    )
    line = re.sub(r"\s+", " ", line)
    return line


def load_text_and_strip(fname):
    """Load text and strip any punctuation"""

    lines = []
    with open(fname, "r", encoding="utf-8") as fpr:
        for line in fpr:
            line = line.strip()
            line = remove_punc(line)
            if line:
                lines.append(line)
    return lines


def load_text(
    fname: Union[str, list],
    subset_ixs=None,
    ignore_ixs=None,
    multi_line=False,
    strip=False,
):
    """Load text line by line. If `mult_line is True` then each entry is
    corresponds to multiple lines until it encounters two  consecutive new-lines"""

    if subset_ixs is not None:
        if isinstance(subset_ixs, np.ndarray):
            subset_ixs = subset_ixs.tolist()
        subset_ixs = set(subset_ixs)

    if ignore_ixs is not None:
        if isinstance(ignore_ixs, np.ndarray):
            ignore_ixs = ignore_ixs.tolist()
        ignore_ixs = set(ignore_ixs)

    fpr = None
    if isinstance(fname, str):
        assert os.path.exists(fname), f"Cannot find the file at {fname}"
        fpr = open(fname, "r", encoding="utf-8")
    elif isinstance(fname, list):
        fpr = fname
    else:
        raise TypeError(f"fname: {fname} should be string (a file path) or list")

    lst = []
    lno = 0
    for line in fpr:
        line = line.strip()
        if line:
            if subset_ixs:
                if lno not in subset_ixs:
                    lno += 1
                    continue

            if ignore_ixs:
                if lno in ignore_ixs:
                    lno += 1
                    continue

            if strip:
                line = remove_punc(line)
            lst.append(line)

        lno += 1

    if isinstance(fname, str):
        fpr.close()

    return lst


def load_bitexts(fpath1, fpath2):
    """Load bitexts from two files, each line is a sentence, empty lines are ignored"""

    text1 = []
    text2 = []
    with (
        open(fpath1, "r", encoding="utf-8") as fpr1,
        open(fpath2, "r", encoding="utf-8") as fpr2,
    ):
        for line1, line2 in zip(fpr1, fpr2):
            line1 = line1.strip()
            line2 = line2.strip()
            if line1 and line2:
                text1.append(line1)
                text2.append(line2)

    return text1, text2


def get_int2vocab(inp: Union[str, CountVectorizer, dict]):
    """Get integer 2 vocab mapping"""

    vocab2int = {}
    int2vocab = {}
    if isinstance(inp, str):
        if inp.endswith("json"):
            vocab2int = {}
            with open(inp, "r", encoding="utf-8") as fpr:
                vocab2int = json.load(fpr)

        else:
            #  ends with .pkl
            with open(inp, "rb") as fpr:
                cvect_obj = pickle.load(fpr)
            vocab2int = cvect_obj.vocabulary_

    elif isinstance(inp, CountVectorizer):
        vocab2int = inp.vocabulary_

    else:
        vocab2int = inp

    for word, idx in vocab2int.items():
        int2vocab[idx] = word

    return int2vocab


def filter_data(
    input_text_file,
    sim_score_file,
    sim_thresh=0.8,
    norm: Union[str, None] = None,
    strip: bool = False,
):
    """
    Filter pairs of input text on similarity threshold and
    select the corresponding input documents (sentences).
    """

    logger.info("Loading text from %s", input_text_file)
    sim_scores = np.load(sim_score_file)
    ixs = np.where(sim_scores > sim_thresh)[0]
    logger.info("Sim scores > %.3f are %d", sim_thresh, len(ixs))

    input_docs = load_text(input_text_file, subset_ixs=ixs, strip=strip)
    logger.info(
        "Input docs (sents) after filtering %d",
        len(input_docs),
    )

    assert len(input_docs) == len(ixs), "Num docs != num ixs"

    return input_docs


def filter_data_and_embs_list(
    list_text_file: list,
    list_emb1_file: list,
    list_emb2_file: list,
    list_sim_file: list,
    sim_thresh: float = 0.0,
    norm: Union[None, str] = None,
    strip: bool = False,
):
    """
    Returns filetered docs, list of memmaps for input embs1 and embs2 and list of indices where emb pairs satisfy sim_thresh constraint.
    """
    all_docs = []
    all_embs1 = []  # list of memmaps
    all_embs2 = []  # list of memmaps
    all_ixs = []  # list of indices that satisfy sim_thresh constraint
    for i, _ in enumerate(list_text_file):
        logger.info(
            "%3d/%3d Loading quadruplet: text from %s",
            (i + 1),
            len(list_text_file),
            list_text_file[i],
        )
        docs, embs1, embs2, ixs = filter_data_and_embs(
            list_text_file[i],
            list_emb1_file[i],
            list_emb2_file[i],
            list_sim_file[i],
            sim_thresh,
            norm=norm,
            strip=strip,
        )
        all_docs.extend(docs)
        all_embs1.append(embs1)
        all_embs2.append(embs2)
        all_ixs.append(ixs)

    # all_embs1 = np.concatenate(all_embs1)
    # all_embs2 = np.concatenate(all_embs2)

    # logger.info("All embs1: %d, %d", *all_embs1.shape)
    # logger.info("All embs2: %d, %d", *all_embs2.shape)
    logger.info(f"All docs : {len(all_docs):,}")

    # assert all_embs1.shape[0] == all_embs2.shape[0], "num rows mismatch in embs1, embs2"
    # assert len(all_docs) == all_embs1.shape[0], "num docs(sents) != num embs"

    return all_docs, all_embs1, all_embs2, all_ixs


def filter_data_and_embs(
    input_text_file,
    input_emb_file1,
    input_emb_file2,
    sim_score_file,
    sim_thresh=0.0,
    sort_order="random",
    norm: Union[str, None] = None,
    strip: bool = False,
):
    """Filter pairs of input embeddings on similarity threshold and select the
    corresponding input documents (sentences).

    Returns filetered docs, memmaps for input embs1 and embs2 and indices where emb pairs  satisfy sim_thresh constraint.
    """

    sim_scores = np.load(sim_score_file)
    logger.info("Similarity scores: %d", sim_scores.shape[0])

    input_embs1 = np.load(input_emb_file1, mmap_mode="r")
    logger.info("Input memmap doc embs1: %d %d", *input_embs1.shape)

    input_embs2 = np.load(input_emb_file2, mmap_mode="r")
    logger.info("Input memmap doc embs2: %d %d", *input_embs2.shape)

    # Sanity checks: ensure all data sources have matching dimensions
    assert (
        input_embs1.shape[0] == input_embs2.shape[0]
    ), f"Num rows mismatch: embs1 ({input_embs1.shape[0]}) != embs2 ({input_embs2.shape[0]})"

    assert (
        sim_scores.shape[0] == input_embs1.shape[0]
    ), f"Num rows mismatch: sim_scores ({sim_scores.shape[0]}) != embs1 ({input_embs1.shape[0]})"

    assert (
        sim_scores.shape[0] == input_embs2.shape[0]
    ), f"Num rows mismatch: sim_scores ({sim_scores.shape[0]}) != embs2 ({input_embs2.shape[0]})"

    assert (
        input_embs1.shape[1] == input_embs2.shape[1]
    ), f"Dim mismatch: embs1 ({input_embs1.shape[1]}) != embs2 ({input_embs2.shape[1]})"

    ixs = np.where(sim_scores > sim_thresh)[0]
    logger.info("Embs with sim scores > %f are %d", sim_thresh, len(ixs))

    input_docs = load_text(input_text_file, subset_ixs=ixs, strip=strip)
    logger.info(
        "Input docs (sents) after selecting subset %d %d",
        len(ixs),
        len(input_docs),
    )

    # Sanity check: ensure we got all the filtered documents
    assert len(input_docs) == len(ixs), (
        f"Num docs mismatch: expected {len(ixs)} docs (from filtered indices) "
        f"but got {len(input_docs)} from text file {input_text_file}. "
        f"Text file may have fewer lines than embeddings."
    )

    return input_docs, input_embs1, input_embs2, ixs


def load_data_and_embs(
    input_text_file,
    input_emb_file1,
    input_emb_file2,
    sim_score_file,
    sim_thresh=0.8,
    sort_order="random",
    norm: Union[str, None] = None,
    strip: bool = False,
):
    """Filter pairs of input embeddings on similarity threshold and select the
    corresponding input documents (sentences).

    Returns filetered docs, embs1, embs2 and indices where emb pairs satisfy sim_thresh constraint.
    """

    # Load similarity scores
    sim_scores = np.load(sim_score_file)
    logger.info("Similarity scores: %d", sim_scores.shape[0])

    # Load full embeddings to check shapes before filtering
    input_embs1_full = np.load(input_emb_file1)
    logger.info("Input doc embs1 (full): %d %d", *input_embs1_full.shape)

    input_embs2_full = np.load(input_emb_file2)
    logger.info("Input doc embs2 (full): %d %d", *input_embs2_full.shape)

    # Sanity checks: ensure all data sources have matching dimensions
    assert (
        input_embs1_full.shape[0] == input_embs2_full.shape[0]
    ), f"Num rows mismatch: embs1 ({input_embs1_full.shape[0]}) != embs2 ({input_embs2_full.shape[0]})"

    assert (
        sim_scores.shape[0] == input_embs1_full.shape[0]
    ), f"Num rows mismatch: sim_scores ({sim_scores.shape[0]}) != embs1 ({input_embs1_full.shape[0]})"

    assert (
        sim_scores.shape[0] == input_embs2_full.shape[0]
    ), f"Num rows mismatch: sim_scores ({sim_scores.shape[0]}) != embs2 ({input_embs2_full.shape[0]})"

    assert (
        input_embs1_full.shape[1] == input_embs2_full.shape[1]
    ), f"Dim mismatch: embs1 ({input_embs1_full.shape[1]}) != embs2 ({input_embs2_full.shape[1]})"

    # Now filter based on similarity threshold
    ixs = np.where(sim_scores > sim_thresh)[0]
    logger.info("Embs with sim scores > %f are %d", sim_thresh, len(ixs))

    input_embs1 = input_embs1_full[ixs, :]
    logger.info("Input doc embs1 (filtered): %d %d", *input_embs1.shape)

    input_embs2 = input_embs2_full[ixs, :]
    logger.info("Input doc embs2 (filtered): %d %d", *input_embs2.shape)

    input_docs = load_text(input_text_file, subset_ixs=ixs, strip=strip)
    logger.info(
        "Input docs (sents) after selecting subset %d %d",
        len(ixs),
        len(input_docs),
    )

    # Sanity check: ensure we got all the filtered documents
    assert len(input_docs) == len(ixs), (
        f"Num docs mismatch: expected {len(ixs)} docs (from filtered indices) "
        f"but got {len(input_docs)} from text file {input_text_file}. "
        f"Text file may have fewer lines than embeddings."
    )

    # get rows where embs1 have NaNs
    nan_ixs1 = np.where(np.isnan(input_embs1).any(axis=1))[0]
    logger.info("Num rows with NaNs in embs1: %d", len(nan_ixs1))
    # get rows where embs2 have NaNs
    nan_ixs2 = np.where(np.isnan(input_embs2).any(axis=1))[0]
    logger.info("Num rows with NaNs in embs2: %d", len(nan_ixs2))

    # delete rows where embs1 or embs2 have NaNs
    nan_ixs = np.asarray(list(set(nan_ixs1) | set(nan_ixs2)))
    logger.info("Num rows with NaNs in embs1 or embs2: %d", len(nan_ixs))
    if len(nan_ixs) > 0:
        input_embs1 = np.delete(input_embs1, nan_ixs, axis=0)
        logger.info("Input doc embs1: %d %d", *input_embs1.shape)
        input_embs2 = np.delete(input_embs2, nan_ixs, axis=0)
        logger.info("Input doc embs2: %d %d", *input_embs2.shape)
        input_docs = [input_docs[i] for i in range(len(input_docs)) if i not in nan_ixs]
        logger.info("Input docs (sents) after removing NaNs %d", len(input_docs))

    assert (
        input_embs1.shape[0] == input_embs2.shape[0]
    ), "Num rows mismatch in embs1, embs2"
    assert len(input_docs) == input_embs1.shape[0], "Num docs(sents) != num embs"
    return input_docs, input_embs1, input_embs2, ixs
