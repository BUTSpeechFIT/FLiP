#!/usr/bin/env python3
"""Train interpretable LoLM/FactLoLM models."""

import sys
import argparse
import os
import json
import pickle
import shutil
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from lolm.utils import create_logger, get_num_params, save_ckpt, move_to_device
from lolm.config import load_train_config
from lolm.data.builders import build_embow_dataset_from_yaml
from lolm.data.datasets import ebow_collator, IntraChunkSampler
from lolm.models.interpretable import LoLM, FactLoLM


@torch.no_grad()
def evaluate_pr(model, device, valid_loader, topk_factor, use_bias=False):
    """Evaluate precision and recall for validation set."""

    max_unique = 100  # max number of unique tokens in a sentence

    scores = {}
    for i in [1, 2]:
        scores[f"val_precision_{i}"] = []
        scores[f"val_recall_{i}"] = []
    scores["val_precision_avg"] = []
    scores["val_recall_avg"] = []

    for batch in valid_loader:
        batch = move_to_device(batch, device)

        for i in [1, 2]:
            if hasattr(model, "rank"):
                # FactLoLM
                if use_bias:
                    logits = model.b + (model.E1 @ model.E2) @ batch[f"doc_embs{i}"].T
                else:
                    logits = (model.E1 @ model.E2) @ batch[f"doc_embs{i}"].T
            else:
                # LoLM
                if use_bias:
                    logits = model.b + model.E @ batch[f"doc_embs{i}"].T
                else:
                    logits = model.E @ batch[f"doc_embs{i}"].T

            _, top_ixs = logits.topk(max_unique, dim=0, sorted=True)

            for j in range(logits.shape[1]):
                v_ixs = batch["cixs"][batch["rixs"] == j]

                if len(v_ixs) > 0:
                    topn = min(max_unique, int(len(v_ixs) * topk_factor))
                    recalled = np.intersect1d(
                        top_ixs[:topn, j].cpu().numpy(), v_ixs.cpu().numpy()
                    )
                    prec_ij = len(recalled) / topn
                    recl_ij = len(recalled) / len(v_ixs)

                    scores[f"val_precision_{i}"].append(prec_ij)
                    scores[f"val_recall_{i}"].append(recl_ij)

    scores["val_num_sentences"] = len(scores["val_precision_1"])
    for i in [1, 2]:
        scores[f"val_precision_{i}"] = round(np.mean(scores[f"val_precision_{i}"]), 4)
        scores[f"val_recall_{i}"] = round(np.mean(scores[f"val_recall_{i}"]), 4)
    scores["val_recall_avg"] = round(
        (scores["val_recall_1"] + scores["val_recall_2"]) / 2, 4
    )
    scores["val_precision_avg"] = round(
        (scores["val_precision_1"] + scores["val_precision_2"]) / 2, 4
    )
    return scores


@torch.no_grad()
def evaluate_ppl(model, device, valid_loader):
    """Evaluate perplexity on validation data."""

    ppls = {}
    for i in range(1, 3):
        ppls[f"val_ppl_{i}"] = 0.0
        ppls[f"val_nllh_{i}"] = 0.0
    ppls["num_toks"] = 0.0

    for batch in valid_loader:
        batch = move_to_device(batch, device)

        num_toks = batch["vals"].sum()
        ppls["num_toks"] += num_toks.item()

        for i in range(1, 3):
            nllh = model.compute_neg_log_likelihood(
                batch["rixs"], batch["cixs"], batch["vals"], batch[f"doc_embs{i}"]
            )
            ppls[f"val_nllh_{i}"] += nllh.clone().item()

    for i in range(1, 3):
        ppls[f"val_ppl_{i}"] = np.exp(ppls[f"val_nllh_{i}"] / ppls["num_toks"])

    return ppls


def train(model, opt, config, device, logger, train_loader, valid_loader, ckpt_dir):
    """Training loop."""

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",
        factor=config["training"]["lr_factor"],
        patience=config["training"]["patience"] // 2,
    )
    patience = config["training"]["patience"]

    # Determine alpha based on target_text position in embedding_pair
    emb_pair = config["data"]["train"]["embedding_pair"]
    target_text = config["data"]["train"]["target_text"]

    # Find which position target_text is in embedding_pair
    if target_text == emb_pair[0]:
        # Target is first embedding (doc_embs1)
        alpha_value = config["model"]["alpha"]
        logger.info(
            "Alpha=%.2f applies to target_text='%s' (embedding_pair[0])",
            alpha_value,
            target_text,
        )
    elif target_text == emb_pair[1]:
        # Target is second embedding (doc_embs2) - flip alpha
        alpha_value = 1.0 - config["model"]["alpha"]
        logger.info(
            "Alpha=%.2f specified for target_text, but '%s' is embedding_pair[1]",
            config["model"]["alpha"],
            target_text,
        )
        logger.info(
            "Using flipped alpha=%.2f for '%s' (doc_embs2)", alpha_value, target_text
        )
    else:
        raise ValueError(
            f"target_text='{target_text}' not found in embedding_pair={emb_pair}"
        )

    alpha = torch.tensor([alpha_value], requires_grad=False).to(device=device)

    logger.info(
        "Embedding weights: %s=%.2f, %s=%.2f",
        emb_pair[0],
        alpha_value,
        emb_pair[1],
        1.0 - alpha_value,
    )

    # Log L1 regularization method
    l1 = config["regularization"]["l1"]
    if l1 > 0:
        logger.info(
            f"L1 regularization: λ={l1}, method={config['regularization']['l1_method']}"
        )

    # Initial validation
    stop_criteria = config["training"]["stop_criteria"]
    val_pr = evaluate_pr(
        model,
        device,
        valid_loader,
        config["evaluation"]["topk_factor"],
        config["evaluation"]["use_bias"],
    )
    logger.info(json.dumps(val_pr))
    best_val = val_pr[f"val_{stop_criteria}"]

    for iteration in range(1, config["training"]["iterations"] + 1):
        for bno, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            opt.zero_grad()

            nllh_1 = model.compute_neg_log_likelihood(
                batch["rixs"], batch["cixs"], batch["vals"], batch["doc_embs1"]
            )

            nllh_2 = model.compute_neg_log_likelihood(
                batch["rixs"], batch["cixs"], batch["vals"], batch["doc_embs2"]
            )

            nllh = (alpha * nllh_1) + ((1.0 - alpha) * nllh_2)

            # L1 Regularization: Subgradient method
            if l1 > 0 and config["regularization"]["l1_method"] == "subgradient":
                l1_penalty = l1 * model.compute_l1_penalty()
                nllh -= l1_penalty

            nllh.backward()
            opt.step()

            # L1 Regularization: Proximal gradient method
            if l1 > 0 and config["regularization"]["l1_method"] == "proximal":
                current_lr = opt.param_groups[0]["lr"]
                model.apply_proximal_operator(l1, current_lr)

            llh_norm = nllh.item() / batch["vals"].sum().item()

            # Compute gradient norm
            grad_norm = torch.sqrt(
                sum(
                    p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
                )
            ).item()

            log_dict = {
                "iter": iteration,
                "batch": bno,
                "llh_1": -1 * round(nllh_1.item(), 4),
                "llh_2": -1 * round(nllh_2.item(), 4),
                "llh_norm": round(llh_norm, 4),
                "loss": round(nllh.item(), 4),
                "lr": "{:.4e}".format(opt.param_groups[0]["lr"]),
                "grad_norm": round(grad_norm, 4),
            }

            # Log sparsity metrics when using L1
            if l1 > 0 and (bno % 100 == 0):
                with torch.no_grad():
                    if hasattr(model, "rank"):
                        total_params = model.E1.numel()
                        zero_params = (model.E1.abs() < 1e-8).sum().item()
                    else:
                        total_params = model.E.numel()
                        zero_params = (model.E.abs() < 1e-8).sum().item()

                    sparsity_pct = 100.0 * zero_params / total_params
                    log_dict["sparsity_pct"] = round(sparsity_pct, 2)
                    log_dict["exact_zeros"] = zero_params

            logger.info(json.dumps(log_dict))

        # Checkpoint and validation
        if (iteration % config["training"]["checkpoint_interval"]) == 0:
            if valid_loader is not None:
                val_pr = evaluate_pr(
                    model,
                    device,
                    valid_loader,
                    config["evaluation"]["topk_factor"],
                    config["evaluation"]["use_bias"],
                )
                val_ppl = evaluate_ppl(model, device, valid_loader)
                val_ppl["iter"] = iteration

                logger.info(json.dumps(val_ppl))
                logger.info(json.dumps(val_pr))

                scheduler.step(val_pr[f"val_{stop_criteria}"])

                if val_pr[f"val_{stop_criteria}"] > best_val:
                    best_val = val_pr[f"val_{stop_criteria}"]
                    save_ckpt(ckpt_dir, f"best_val_{stop_criteria}", model, opt)
                    patience = config["training"]["patience"]
                else:
                    if config["training"]["early_stopping"]:
                        if patience == 0:
                            logger.info(
                                "No patience left. Early stopping at iter %d, "
                                "since val_%s did not improve.",
                                iteration,
                                stop_criteria,
                            )
                            save_ckpt(ckpt_dir, iteration, model, opt)
                            return model, opt
                        else:
                            logger.info("Patience left %d", patience)
                            patience -= 1
            else:
                save_ckpt(ckpt_dir, iteration, model, opt)

            model.cuda()

    return model, opt


def main():
    """Main training function."""

    args = parse_arguments()

    # Load configuration with CLI overrides
    config = load_train_config(args.config, cli_args=args)

    # Load dataset configs (support both old and new format)
    train_data_yaml = config["data"].get("train_datasets_yaml") or config["data"].get(
        "datasets_yaml"
    )
    dev_data_yaml = config["data"].get("dev_datasets_yaml", train_data_yaml)

    # train_data_config = load_data_config(train_data_yaml)
    # dev_data_config = (
    #     load_data_config(dev_data_yaml)
    #     if dev_data_yaml != train_data_yaml
    #     else train_data_config
    # )

    # vocab_config = load_vocab_config(config["data"]["vocab_yaml"])

    # Set random seeds
    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    # Setup experiment directory
    exp_dir = os.path.join(
        config["experiment"]["output_dir"], config["experiment"]["name"]
    )
    os.makedirs(exp_dir, exist_ok=True)

    # Check if experiment already exists
    config_path = os.path.join(exp_dir, "train_config.yaml")
    if os.path.exists(config_path) and not args.ovr:
        print(
            f"ERROR: Experiment directory already exists: {exp_dir}\n"
            f"Found existing train_config.yaml. This would overwrite the previous experiment.\n"
            f"Either:\n"
            f"  1. Use a different experiment name (--name <new_name>)\n"
            f"  2. Pass --ovr flag to allow overwriting\n"
            f"  3. Manually remove/rename the existing experiment directory"
        )
        sys.exit(1)

    # Create subdirectories for organization
    logs_dir = os.path.join(exp_dir, "logs")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create logger
    logger = create_logger(
        os.path.join(logs_dir, "train"), verbose=config["compute"]["verbose"]
    )

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "Not_set")
        if device_id == "Not_set":
            logger.warning("CUDA_VISIBLE_DEVICES not set")
        logger.info("CUDA_VISIBLE_DEVICES=%s", device_id)

    # Save configuration to experiment directory
    with open(os.path.join(exp_dir, "train_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config to %s", os.path.join(exp_dir, "train_config.yaml"))

    logger.info("Configuration:\n%s", json.dumps(config, indent=2))

    # Load CountVectorizer
    vocab_dir = os.path.dirname(config["data"]["vocab_yaml"])
    cvect_path = os.path.join(vocab_dir, "cvect.pkl")

    if not os.path.exists(cvect_path):
        logger.error("CountVectorizer not found at %s", cvect_path)
        sys.exit(1)

    logger.info("Loading CountVectorizer from %s", cvect_path)
    with open(cvect_path, "rb") as f:
        cvect = pickle.load(f)

    # Setup tmp directory if requested
    if config["compute"]["copy_to_tmp"]:
        import tempfile

        os.makedirs("/tmp/" + os.environ["USER"], exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix="lolm_", dir="/tmp/" + os.environ["USER"])
        logger.info("Using tmp dir: %s", tmp_dir)
    else:
        tmp_dir = config["compute"].get("tmp_dir")

    # Build training dataset
    logger.info("Building training dataset")
    logger.info("Using training datasets from: %s", train_data_yaml)
    train_dset = build_embow_dataset_from_yaml(
        data_yaml=train_data_yaml,
        vocab_yaml_or_cvect=cvect,
        dataset_names=config["data"]["train"]["datasets"],
        emb_pair=config["data"]["train"]["embedding_pair"],
        target_text_id=config["data"]["train"]["target_text"],
        sim_thresh=config["data"]["train"]["sim_threshold"],
        norm=config["data"]["train"].get("normalize"),
        tmp_dir=tmp_dir,
        exp_dir=exp_dir,
    )

    logger.info("Training dataset: %d samples", len(train_dset))
    logger.info("Vocabulary size: %d", train_dset.vocab_size)
    logger.info("Embedding dimension: %d", train_dset.emb_dim)

    # Build validation dataset
    valid_dset = None
    valid_loader = None
    if "validation" in config["data"]:
        logger.info("Building validation dataset")
        logger.info("Using validation datasets from: %s", dev_data_yaml)
        valid_dset = build_embow_dataset_from_yaml(
            data_yaml=dev_data_yaml,
            vocab_yaml_or_cvect=cvect,
            dataset_names=config["data"]["validation"]["datasets"],
            emb_pair=config["data"]["validation"]["embedding_pair"],
            target_text_id=config["data"]["validation"]["target_text"],
            sim_thresh=config["data"]["validation"]["sim_threshold"],
            norm=config["data"]["validation"].get("normalize"),
            tmp_dir=tmp_dir,
        )
        logger.info("Validation dataset: %d samples", len(valid_dset))

        valid_sampler = IntraChunkSampler(
            valid_dset.chunk_sizes, config["training"]["batch_size"], shuffle=False
        )
        valid_loader = DataLoader(
            valid_dset, batch_sampler=valid_sampler, collate_fn=ebow_collator
        )

    # Build model
    if config["model"]["type"] == "LoLM":
        model = LoLM(train_dset.vocab_size, train_dset.emb_dim)
        logger.info("Created LoLM model (full rank)")
    elif config["model"]["type"] == "FactLoLM":
        rank = config["model"]["rank"]
        if rank > train_dset.emb_dim:
            logger.error(
                f"Rank ({rank}) must be <= embedding dimension ({train_dset.emb_dim})"
            )
            sys.exit(1)
        model = FactLoLM(train_dset.vocab_size, train_dset.emb_dim, rank)
        logger.info("Created FactLoLM model with rank %d", rank)
    else:
        logger.error("Unknown model type: %s", config["model"]["type"])
        sys.exit(1)

    # Initialize bias
    model.init_bias_with_log_unigram_dist(train_dset.dbyw)

    # Resume from checkpoint if specified
    resume_cfg = config.get("resume", {})
    if resume_cfg.get("model_checkpoint"):
        logger.info("Loading model checkpoint: %s", resume_cfg["model_checkpoint"])
        model.load_state_dict(torch.load(resume_cfg["model_checkpoint"]))

    # Move to device and compile
    model.to(device)
    logger.info("Model on %s", device)
    model = torch.compile(model)

    # Log model info
    logger.info("%s", model)
    num_params, trainable = get_num_params(model)
    logger.info("Num params %.2f M, trainable %.2f M", num_params, trainable)

    # Create optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["regularization"]["weight_decay"],
    )

    # Resume optimizer if specified
    if resume_cfg.get("optimizer_checkpoint"):
        if resume_cfg["optimizer_checkpoint"] == "auto":
            # Derive from model checkpoint
            if resume_cfg.get("model_checkpoint"):
                opt_path = resume_cfg["model_checkpoint"].replace(
                    "model_state_dict_", "optim_state_dict_"
                )
                if os.path.exists(opt_path):
                    logger.info("Loading optimizer checkpoint: %s", opt_path)
                    opt.load_state_dict(torch.load(opt_path))
        else:
            logger.info(
                "Loading optimizer checkpoint: %s", resume_cfg["optimizer_checkpoint"]
            )
            opt.load_state_dict(torch.load(resume_cfg["optimizer_checkpoint"]))

    # Save cvect to exp_dir
    with open(os.path.join(exp_dir, "cvect.pkl"), "wb") as fpw:
        pickle.dump(cvect, fpw)
    logger.info("Saved cvect.pkl to %s", exp_dir)

    # Create data loaders
    train_sampler = IntraChunkSampler(
        train_dset.chunk_sizes, config["training"]["batch_size"], shuffle=False
    )
    train_loader = DataLoader(
        train_dset,
        collate_fn=ebow_collator,
        pin_memory=True,
        batch_sampler=train_sampler,
        num_workers=config["training"]["num_workers"],
        prefetch_factor=(
            config["training"]["num_workers"]
            if config["training"]["num_workers"] > 0
            else None
        ),
    )

    # Train
    err = None
    try:
        model, opt = train(
            model, opt, config, device, logger, train_loader, valid_loader, ckpt_dir
        )
        save_ckpt(ckpt_dir, config["training"]["iterations"], model, opt)
        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.error("Keyboard interrupt")
        save_ckpt(ckpt_dir, "interrupt", model, opt)

    except RuntimeError as e:
        logger.error("Runtime error: %s", e)
        err = e

    finally:
        # Cleanup tmp directory if used
        if config["compute"]["copy_to_tmp"]:
            logger.info("Removing tmp dir %s", tmp_dir)
            shutil.rmtree(tmp_dir)

        if err:
            raise err


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config", required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--ovr",
        action="store_true",
        help="Allow overwriting existing experiment directory",
    )

    # Config overrides group - most commonly adjusted parameters
    override_group = parser.add_argument_group(
        "config overrides",
        "Override specific config values from command line (overrides YAML settings)",
    )

    # Experiment settings
    override_group.add_argument(
        "--name", type=str, help="Experiment name (overrides experiment.name)"
    )
    override_group.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (overrides experiment.output_dir)",
    )
    override_group.add_argument(
        "--seed", type=int, help="Random seed (overrides experiment.seed)"
    )

    # Training hyperparameters
    override_group.add_argument(
        "--lr", type=float, help="Learning rate (overrides training.learning_rate)"
    )
    override_group.add_argument(
        "--iterations",
        type=int,
        help="Number of training iterations (overrides training.iterations)",
    )
    override_group.add_argument(
        "--batch_size", type=int, help="Batch size (overrides training.batch_size)"
    )

    # Regularization
    override_group.add_argument(
        "--l1",
        type=float,
        help="L1 regularization coefficient (overrides regularization.l1)",
    )
    override_group.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay / L2 regularization (overrides regularization.weight_decay)",
    )

    # Model
    override_group.add_argument(
        "--alpha",
        type=float,
        help="Alpha for combining objectives (overrides model.alpha)",
    )
    override_group.add_argument(
        "--rank",
        type=int,
        help="Rank for FactLoLM (overrides model.rank, only used for FactLoLM)",
    )

    # Resume training
    override_group.add_argument(
        "--resume_model",
        type=str,
        help="Path to model checkpoint (overrides resume.model_checkpoint)",
    )
    override_group.add_argument(
        "--resume_optim",
        type=str,
        help="Path to optimizer checkpoint or 'auto' (overrides resume.optimizer_checkpoint)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
