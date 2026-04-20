"""Configuration loaders and validators for YAML configs."""

import os
import yaml
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_config(path: str, cli_args=None) -> Dict[str, Any]:
    """Load and validate training configuration.

    Args:
        path: Path to YAML config file
        cli_args: Optional argparse.Namespace with CLI overrides

    Returns:
        Configuration dictionary with CLI overrides applied
    """
    config = load_yaml(path)

    # Apply CLI overrides before validation
    if cli_args is not None:
        config = apply_cli_overrides(config, cli_args)

    validate_train_config(config)
    return config


def load_vocab_config(path: str) -> Dict[str, Any]:
    """Load vocabulary configuration."""
    return load_yaml(path)


def load_data_config(path: str) -> Dict[str, Any]:
    """Load data configuration (datasets.yaml)."""
    return load_yaml(path)


def apply_cli_overrides(config: Dict[str, Any], cli_args) -> Dict[str, Any]:
    """Apply command line argument overrides to config.

    Args:
        config: Configuration dictionary from YAML
        cli_args: argparse.Namespace with CLI arguments

    Returns:
        Updated config dictionary
    """
    # Experiment overrides
    if cli_args.name is not None:
        config.setdefault("experiment", {})["name"] = cli_args.name

    if cli_args.output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = cli_args.output_dir

    if cli_args.seed is not None:
        config.setdefault("experiment", {})["seed"] = cli_args.seed

    # Training overrides
    if cli_args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = cli_args.lr

    if cli_args.iterations is not None:
        config.setdefault("training", {})["iterations"] = cli_args.iterations

    if cli_args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = cli_args.batch_size

    # Regularization overrides
    if cli_args.l1 is not None:
        config.setdefault("regularization", {})["l1"] = cli_args.l1

    if cli_args.weight_decay is not None:
        config.setdefault("regularization", {})["weight_decay"] = cli_args.weight_decay

    # Model overrides
    if cli_args.alpha is not None:
        config.setdefault("model", {})["alpha"] = cli_args.alpha
    if cli_args.rank is not None:
        config.setdefault("model", {})["rank"] = cli_args.rank

    # Resume overrides
    if cli_args.resume_model is not None:
        config.setdefault("resume", {})["model_checkpoint"] = cli_args.resume_model

    if cli_args.resume_optim is not None:
        config.setdefault("resume", {})["optimizer_checkpoint"] = cli_args.resume_optim

    return config


def _convert_numeric_fields(config: Dict[str, Any]) -> None:
    """Convert numeric string fields to proper types in-place.

    Handles scientific notation (e.g., '1e-4') that PyYAML may parse as strings.

    Args:
        config: Configuration dictionary to modify in-place

    Raises:
        ValueError: If conversion fails for expected numeric fields
    """
    # Define fields that should be numeric: (section_path, field_name, type)
    numeric_conversions = [
        # Training parameters
        (["training"], "learning_rate", float),
        (["training"], "lr_factor", float),
        (["training"], "batch_size", int),
        (["training"], "iterations", int),
        (["training"], "num_workers", int),
        (["training"], "checkpoint_interval", int),
        (["training"], "patience", int),
        # Regularization parameters
        (["regularization"], "l1", float),
        (["regularization"], "weight_decay", float),
        # Model parameters
        (["model"], "alpha", float),
        (["model"], "rank", int),
        # Evaluation parameters
        (["evaluation"], "topk_factor", float),
        # Experiment parameters
        (["experiment"], "seed", int),
        # Data parameters
        (["data", "train"], "sim_threshold", float),
        (["data", "validation"], "sim_threshold", float),
    ]

    for path, field, dtype in numeric_conversions:
        # Navigate to the section
        section = config
        try:
            for key in path:
                if key not in section:
                    break
                section = section[key]
            else:
                # Section exists, try to convert the field if present
                if field in section and section[field] is not None:
                    try:
                        section[field] = dtype(section[field])
                    except (ValueError, TypeError) as e:
                        field_path = ".".join(path + [field])
                        raise ValueError(
                            f"Invalid value for {field_path}: '{section[field]}' "
                            f"(expected {dtype.__name__})"
                        ) from e
        except (TypeError, KeyError):
            # Section doesn't exist or isn't a dict, skip
            continue


def validate_train_config(config: Dict[str, Any]) -> None:
    """Validate training configuration structure.

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Convert numeric strings first (handles scientific notation)
    _convert_numeric_fields(config)

    required_sections = ["experiment", "data", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate experiment section
    exp = config["experiment"]
    if "name" not in exp:
        raise ValueError("experiment.name is required")
    if "output_dir" not in exp:
        raise ValueError("experiment.output_dir is required")

    # Validate data section
    data = config["data"]
    if "train_datasets_yaml" not in data and "datasets_yaml" not in data:
        raise ValueError(
            "data.train_datasets_yaml (or)  data.datasets_yaml is required"
        )
    if "vocab_yaml" not in data:
        raise ValueError("data.vocab_yaml is required")
    if "train" not in data:
        raise ValueError("data.train is required")

    # Validate train data config
    train = data["train"]
    required_train_fields = ["datasets", "embedding_pair", "target_text"]
    for field in required_train_fields:
        if field not in train:
            raise ValueError(f"data.train.{field} is required")

    if len(train["embedding_pair"]) != 2:
        raise ValueError("data.train.embedding_pair must have exactly 2 elements")

    # Validate model section
    model = config["model"]
    if "type" not in model:
        raise ValueError("model.type is required")
    if model["type"] not in ["LoLM", "FactLoLM"]:
        raise ValueError(
            f"model.type must be 'LoLM' or 'FactLoLM', got: {model['type']}"
        )
    if model["type"] == "FactLoLM" and "rank" not in model:
        raise ValueError("model.rank is required for FactLoLM")

    # Validate training section
    training = config["training"]
    required_training_fields = ["batch_size", "learning_rate", "iterations"]
    for field in required_training_fields:
        if field not in training:
            raise ValueError(f"training.{field} is required")

    # Validate target_text is in embedding_pair
    emb_pair = config["data"]["train"]["embedding_pair"]
    target_text = config["data"]["train"]["target_text"]

    if target_text not in emb_pair:
        raise ValueError(
            f"In configuration"
            f"target_text='{target_text}' must be one of embedding_pair={emb_pair}\n"
            f"Fix the YAML configuration file."
        )


def merge_configs(train_config: Dict, data_config: Dict, vocab_config: Dict) -> Dict:
    """Merge all configurations into a single dict.

    This is useful for having all configuration in one place.
    """
    return {
        "train": train_config,
        "data": data_config,
        "vocab": vocab_config,
    }
