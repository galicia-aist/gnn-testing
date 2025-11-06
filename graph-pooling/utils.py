import os
import random
from datetime import datetime

import numpy as np
import torch
from itertools import chain  # 2024.7.31

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import CitationFull, Amazon, Actor, Reddit

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