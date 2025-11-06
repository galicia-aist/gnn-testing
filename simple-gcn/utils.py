import argparse
import json
import logging
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import NeighborLoader



def get_args():
    parser = argparse.ArgumentParser()

    # Core experiment setup
    parser.add_argument('--model', type=str, choices=['GCN', 'SGC'], required=True, help='Instance or Embedding model')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Cora', 'CiteSeer', 'PubMed', 'CoraFull', 'CiteseerFull', 'PubMedFull', 'Reddit', 'Products', 'Arxiv', 'Mag'],
                        help='Dataset to use')
    # Training configuration
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--bag_ratio', type=float, default=0.8, help='Inner bag ratio of positive and negative labels')

    # Execution setup
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA training (default: False).')
    parser.add_argument('--loaders', action='store_true', help='Enable NeighborLoader for mini-batch training (default: False).')
    parser.add_argument('--ddp', action='store_true', default=False, help='Use Distributed Data Parallelism.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')

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


def get_loaders(args, data, world_size=1, rank=0, logger=None):
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

    if args.loaders:
        def make_loader(mask, neighbors, shuffle, sampler=None):
            return NeighborLoader(
                data,
                num_neighbors=neighbors,
                input_nodes=mask,
                batch_size=1024,
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

            train_loader = make_loader(data.train_mask, [-1, -1], True, make_sampler(data.train_mask, True))
            val_loader = make_loader(data.val_mask, [25, 10], True, make_sampler(data.val_mask, True))
            test_loader = make_loader(data.test_mask, [25, 10], False, make_sampler(data.test_mask, False))

        else:
            # --- Regular loaders ---
            train_loader = make_loader(data.train_mask, [10, 5], True)
            val_loader = make_loader(data.val_mask, [25, 10], False)
            test_loader = make_loader(data.test_mask, [25, 10], False)

        logger.debug(f"Train loader batches: {len(train_loader)}")
        logger.debug(f"Validation loader batches: {len(val_loader)}")
        logger.debug(f"Test loader batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    return None, None, None