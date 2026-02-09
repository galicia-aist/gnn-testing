import os
import random
from datetime import datetime

import numpy as np
import torch
from itertools import chain  # 2024.7.31

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CitationFull, Amazon, Actor, Reddit
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import NeighborLoader

import argparse
import json
import logging


@torch.no_grad()
def eval_metrics(out, nodes, labels):
    y_pred = (out[nodes] > 0).long()
    acc = (y_pred == labels.long()).float().mean().item()
    return acc

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def get_flattened_nodes(bag_node, chosen_label, labels, r_start, r_end):
    flattened_nodes = dict()
    bag_node_keys = list(bag_node.keys())
    bag_node_keys = bag_node_keys[r_start:r_end]
    # https://stackoverflow.com/questions/9401209/how-to-merge-an-arbitrary-number-of-tuples-in-python
    node_in_bags = list(chain.from_iterable(bag_node_keys))
    node_in_bags = list(set(node_in_bags))
    random.shuffle(node_in_bags)

    for node_id in node_in_bags:
        if labels[node_id].item() == chosen_label:
            flattened_nodes[tuple([node_id])] = 1
        else:
            flattened_nodes[tuple([node_id])] = 0

    return flattened_nodes

def load_data(d):
    """Load citation network dataset (cora only for now)"""
    if d.lower() == "coraml":
        dataset = CitationFull(root='../datasets/citationfull/', name='Cora_ML')
    elif d.lower() == "citeseer":
        dataset = CitationFull(root='../datasets/citationfull/', name='CiteSeer')
    elif d.lower() == "photo":
        dataset = Amazon(root='../datasets/amazon/', name='photo')
    elif d.lower() == "actor":
        dataset = Actor(root='../datasets/actor/')
    elif d.lower() == "reddit":
        dataset = Reddit(root='../datasets/reddit/')
    elif d.lower() in {"arxiv", "products", "mag"}:
        dataset = PygNodePropPredDataset(name=f"ogbn-{d.lower()}", root='../datasets/')
    device = torch.device('cpu')
    dataset = dataset[0].to(device)

    return dataset

def write_experiment_summary(args, test_results, total_time, filename=None):
    """
    Write a formatted experiment summary to a file.

    Args:
        args: argparse.Namespace containing experiment arguments
        test_results: list of dicts with keys {epoch, loss, accuracy, auc}
        total_time: float, total training time in seconds
        filename: optional string, path to output file. If None, will auto-generate.
    """
    # Create results directory if not exists
    os.makedirs("results", exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"results/{args.model}_{args.d}_MIL{args.mode}_{args.pool}_{args.class_type}_{timestamp}.txt"

    # Identify best results
    best_acc_result = max(test_results, key=lambda x: x["accuracy"])
    best_auc_result = max(test_results, key=lambda x: x["auc"])

    with open(filename, "w") as f:
        f.write("===== Experiment Summary =====\n\n")

        # Arguments
        f.write(">>> Arguments used:\n")
        max_len = max(len(k) for k in vars(args).keys())
        for k, v in vars(args).items():
            f.write(f"  {k.ljust(max_len)} : {v}\n")
        f.write("\n")

        # Training info
        f.write(f">>> Total training time: {total_time:.2f} seconds\n\n")

        # Results per checkpoint
        f.write(">>> Test results (every 100 epochs):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':>6} | {'Loss':>10} | {'Accuracy':>10} | {'AUC':>10} | {'Time(s)':>10}\n")
        f.write("-" * 80 + "\n")
        for r in test_results:
            f.write(f"{r['epoch']:>6} | {r['loss']:>10.4f} | {r['accuracy']:>10.4f} | "
                    f"{r['auc']:>10.4f} | {r['time']:>10.2f}\n")
        f.write("-" * 80 + "\n\n")


        # Best epochs
        f.write(">>> Best results:\n")
        f.write(f"  Highest accuracy at epoch {best_acc_result['epoch']} "
                f"(acc={best_acc_result['accuracy']:.4f}, auc={best_acc_result['auc']:.4f})\n")
        f.write(f"  Highest AUC at epoch {best_auc_result['epoch']} "
                f"(auc={best_auc_result['auc']:.4f}, acc={best_auc_result['accuracy']:.4f})\n")

    return filename

def get_args():
    """
    Builds and parses command-line arguments from a JSON configuration file
    (`args-config.json`). It reads each argument’s properties (such as type, default, help text, and flags) from the
     file, adds them to an `ArgumentParser`, and returns the parsed arguments, allowing flexible and easily
     maintainable experiment setups without changing the code.
    """

    with open("args-config.json", 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()

    for arg, props in config.items():
        kwargs = {}

        # Add common fields
        if "help" in props:
            kwargs["help"] = props["help"]
        if "default" in props:
            kwargs["default"] = props["default"]
        if "choices" in props:
            kwargs["choices"] = props["choices"]

        # Determine type
        if "type" in props:
            type_map = {"int": int, "float": float, "str": str, "bool": bool}
            kwargs["type"] = type_map.get(props["type"], str)

        # Handle flags
        if "action" in props:
            kwargs["action"] = props["action"]

        # Handle required flag
        if props.get("required", False):
            kwargs["required"] = True

        parser.add_argument(f'--{arg}', **kwargs)

    args = parser.parse_args()

    return args


class CustomFormatter(logging.Formatter):
    """Custom formatter to include the current GPU in log messages with colors."""

    # ANSI color codes
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Log format with GPU info
    format = "%(asctime)s - %(gpu_info)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    # Different colors for different log levels
    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        # Get current GPU info
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            record.gpu_info = f"GPU: {gpu_id} ({gpu_name})"
        else:
            record.gpu_info = "GPU: CPU"

        # Select the appropriate format based on log level
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger():
    """Sets up the logger with GPU info and color-coded formatting."""
    with open("global_settings.json", "r") as file:
        loaded_data = json.load(file)

    logger_settings = loaded_data["logger"]
    model = logger_settings["model"]
    log_path = logger_settings["log_path"]
    dataset_name = logger_settings["dataset"]
    log_level = logger_settings["log_level"]

    if log_path == "local":
        logs_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_path = f"{logs_dir}//{model}_dataset{dataset_name}.log"

    logger = logging.getLogger(model)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set the logger level dynamically
        logger.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)

    return logger


def log_experiment_settings(logger, args):
    """
    Logs all experiment settings in a nicely formatted table as a single log entry,
    including the PyTorch version.

    Args:
        logger: The logger instance to use.
        args: An argparse.Namespace or any object with attributes to log.
    """
    # Find the longest argument name for alignment
    max_len = max(len(k) for k in vars(args).keys())

    # Build the table as a string
    lines = []
    lines.append("-" * (max_len + 30))
    lines.append(f"{'Argument'.ljust(max_len)} | Value")
    lines.append("-" * (max_len + 30))

    for k, v in vars(args).items():
        lines.append(f"{k.ljust(max_len)} | {v}")

    # Add PyTorch version
    lines.append(f"{'torch_version'.ljust(max_len)} | {torch.__version__}")

    lines.append("-" * (max_len + 30))

    # Join everything into a single string and log once
    logger.info("Experiment settings:\n" + "\n".join(lines))


def get_loaders(args, train_mask, val_mask, test_mask, world_size=1, rank=0, logger=None):
    """
    Creates NeighborLoaders with or without DistributedSamplers depending on args.ddp.

    Args:
        args: parsed command-line arguments (expects args.ddp and args.d)
        data: PyG dataset object
        idx_train, idx_val, idx_test: node indices for each split
        world_size: number of processes (GPUs)
        rank: current process rank

    Returns:
        train_loader, val_loader, test_loader
    """
    def make_loader(mask, neighbors, shuffle, sampler=None):
        return NeighborLoader(
            data,
            num_neighbors=neighbors,
            input_nodes=mask,
            batch_size=4096,
            shuffle=shuffle,
            sampler=sampler
        )

    if getattr(args, "ddp", False):
        # --- Distributed samplers ---
        def make_sampler(mask, shuffle):
            return DistributedSampler(
                mask.nonzero().squeeze(),
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle
            )

        train_loader = make_loader(train_mask, [10, 5], True, make_sampler(data.train_mask, True))
        val_loader = make_loader(val_mask, [25, 10], True, make_sampler(data.val_mask, True))
        test_loader = make_loader(test_mask, [25, 10], False, make_sampler(data.test_mask, False))

    else:
        # --- Regular loaders ---
        train_loader = make_loader(train_mask, [25, 10], True)
        val_loader = make_loader(val_mask, [25, 10], False)
        test_loader = make_loader(test_mask, [25, 10], False)

        # Get the root logger
        root_logger = logging.getLogger()

        # Remove any handlers attached to the root logger
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        logger.debug(f"Train loader batches: {len(train_loader)}")
        logger.debug(f"Validation loader batches: {len(val_loader)}")
        logger.debug(f"Test loader batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

def make_input_nodes(node_ids, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[node_ids] = True
    return mask

def split_labels(labels, num_class):
    """
    Splits node indices into label-wise dictionaries of nodes and non-nodes.

    Args:
        labels (torch.Tensor): 1D tensor of node labels.

    Returns:
        label_node (dict): Keys are label IDs, values are lists of node indices with that label.
        label_non_node (dict): Keys are label IDs, values are lists of node indices NOT having that label.
    """

    label_node = {i: [] for i in range(num_class)}
    label_non_node = {i: [] for i in range(num_class)}

    # Assign nodes to their labels
    for i in range(labels.shape[0]):
        label_id = labels[i].item()
        label_node[label_id].append(i)

    # Assign nodes to non-label lists
    for i in range(num_class):
        for j in range(num_class):
            if j != i:
                label_non_node[i] += label_node[j]

    return label_node, label_non_node