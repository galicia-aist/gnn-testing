import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from itertools import chain  # 2024.7.31

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CitationFull, Amazon, Actor, Reddit


def get_args():
    parser = argparse.ArgumentParser()

    # Core experiment setup
    parser.add_argument('--model', type=str, choices=['gcn', 'gat', 'sage'], required=True, help='Choose model from GCN, GAT, SAGE')
    parser.add_argument('--d', type=str, choices=['coraml', 'citeseer', 'photo', 'actor', 'reddit', 'arxiv'], required=True,
                        help='Dataset to use (coraml=Cora-ML, citeseer=CiteSeer, photo=Amazon-Photo, actor=Actor, reddit=Reddit, arxiv=OGB Arxiv).')
    parser.add_argument('--mode', type=int, required=True,
                        help='Mode parameter. 1 for collective MIL. 2 for standard MIL.')
    parser.add_argument('--class_type', type=str, choices=['node', 'bag'], required=True, help='Classification type')
    parser.add_argument('--pool', type=str, choices=['sum', 'mean', 'max', 'attention', 'set2set', 'default'], required=True,
                        help='Graph pooling type.')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--bag_ratio', type=float, default=0.8, help='Inner bag ratio of positive and negative labels')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    # Execution setup
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA training (default: False).')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--ddp', action='store_true', default=False, help='Use Distributed Data Parallelism.')

    # Reproducibility & logging
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'], help='Logging level.')
    parser.add_argument('--log_path', type=str, default='local', help='Path to store logs. Default is "local".')

    # Extra hyperparameters
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter.')
    parser.add_argument('--rho', type=float, default=1.0, help='Rho parameter.')

    args = parser.parse_args()

    return args

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
    elif d.lower() == "arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='./datasets/')
    device = torch.device('cpu')
    dataset = dataset[0].to(device)

    return dataset

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