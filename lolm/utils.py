import json
import gzip
import os
from glob import glob
import sys
import logging
import platform
import torch
from numpyencoder import NumpyEncoder
from safetensors.torch import save_file


def move_to_device(batch: dict, device) -> dict:
    """Move batch dict of tensors to the given device"""
    new_batch = {}
    for key, elem in batch.items():
        new_batch[key] = elem.to(device)
    return new_batch


def save_json_gzip(fname, data):
    """Save json in compressed gzip"""
    with gzip.open(fname, "w") as fout:
        fout.write(
            json.dumps(data, ensure_ascii=False, cls=NumpyEncoder).encode("utf-8")
        )


def load_json_gzip(fname):
    """Load gzipped json and return"""

    data = {}
    with gzip.open(fname, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
    return data


def save_ckpt(out_dir, sfx, model, opt):
    """Save model checkpoint as safetensors and optimizer state as .pt."""

    # Unwrap compiled model before saving
    raw = getattr(model, "_orig_mod", model)
    save_file(
        raw.state_dict(), os.path.join(out_dir, f"model_state_dict_{sfx}.safetensors")
    )
    torch.save(opt.state_dict(), os.path.join(out_dir, f"optim_state_dict_{sfx}.pt"))


def load_model_ckpt(path, device="cpu"):
    """Load a model state_dict from either .safetensors or legacy .pt format."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(device))
    return torch.load(path, map_location=device, weights_only=True)


def get_num_params(model):
    """Get total number of params and trainable params."""
    req_grad = 0
    non_grad = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            req_grad += param.numel()
        else:
            non_grad += param.numel()
    return ((non_grad + req_grad) / 1e6), (req_grad / 1e6)


def create_logger(log_file_base: str, verbose: bool):
    """Create logger."""

    os.makedirs(os.path.dirname(log_file_base), exist_ok=True)

    if os.path.exists(log_file_base + ".log"):
        num = glob(log_file_base + "*.log")
        os.rename(log_file_base + ".log", f"{log_file_base}.{len(num)}.log")

    logging.basicConfig(
        format="%(levelname)-8s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        filename=f"{log_file_base}.log",
        level=logging.INFO,
        filemode="w",
    )
    print(f"Log file {log_file_base}.log")
    logger = logging.getLogger()
    stdout = logging.StreamHandler(stream=sys.stdout)
    if verbose:
        stdout.setLevel(logging.INFO)
    else:
        stdout.setLevel(logging.WARNING)
    logger.addHandler(stdout)

    logger.info(" ".join(sys.argv))
    logger.info("%s", platform.node())
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "Not_set")
    )

    return logger


def init_exp(args):
    """Set random seed, and check for CUDA (LEGACY - for backward compatibility)."""
    import numpy as np
    import tempfile

    assert (
        args.ckpt_inter < args.iters
    ), f"Checkpoint interval {args.ckpt_inter} must be < training iterations {args.iters}"

    torch.manual_seed(0)
    np.random.seed(args.seed)

    if args.cuda:
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "Not_set")
        if device == "Not_set":
            print(
                "CUDA_VISIBLE_DEVICES env variable is not set. Set it or pass --nocuda argument."
            )
            sys.exit()

    if args.copy_to_tmp:
        # make temp dir inside username subdir
        os.makedirs("/tmp/" + os.environ["USER"], exist_ok=True)
        args.tmp_dir = tempfile.mkdtemp(
            prefix="lolm_", dir="/tmp/" + os.environ["USER"]
        )


def setup_lolm_exp(args):
    """Set up exp dir, create logger, etc (LEGACY - for backward compatibility)."""
    import pickle
    import json
    import numpy as np

    init_exp(args)

    if not args.exp_dir:
        prefix = f"norm_{args.norm}_thr_{args.sim_thresh}_target_{args.target_text_idx}"
        if "mono_index" in args.__dict__:
            if args.mono_index > 0:
                prefix += f"_mono_{args.mono_index}"

        assert os.path.isfile(args.cvect_pkl)

        with open(args.cvect_pkl, "rb") as fpr:
            cvect = pickle.load(fpr)

        args.ngram_range = list(cvect.ngram_range)
        args.min_df = cvect.min_df
        args.max_vocab = (
            len(cvect.vocabulary_) if not cvect.max_features else cvect.max_features
        )

        ngram_str = "mv_{:d}_ngram_{:d}_{:d}_mindf_{:d}".format(
            args.max_vocab,
            *args.ngram_range,
            args.min_df,
        )

        rank = "full" if args.rank == -1 else args.rank

        exp_dir = os.path.join(
            args.out_dir,
            f"{prefix}_{ngram_str}/rank_{rank}_alpha_{args.alpha:.1f}_l1_{args.l1}_wdecay_{args.wdecay}_lr_{args.lr}_{args.stop_criteria}_{args.topk_factor:.1f}",
        )
        args.exp_dir = exp_dir
    else:
        exp_dir = args.exp_dir

    os.makedirs(exp_dir, exist_ok=True)

    phase = "train"
    if args.sdict:
        phase = "resume_train"

    log_file_base = os.path.join(exp_dir, phase)

    logger = create_logger(log_file_base, args.verbose)
    logger.info("%s", json.dumps(args.__dict__))

    with open(os.path.join(exp_dir, "args.json"), "w", encoding="utf-8") as fpw:
        json.dump(args.__dict__, fpw, indent=2)

    return logger
