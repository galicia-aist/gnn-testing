import math
import random
from itertools import chain  # 2024.7.31
import torch
import torch.nn.functional as F
import time
from torch_geometric.datasets import CitationFull, Amazon, Actor
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)  # one logit for binary classification
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(-1)


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

def get_eval_nodes(m, n, chosen_label, labels, label_node, label_non_node, seed=42):
    bag_node = dict()
    random.seed(seed)
    torch.manual_seed(seed)

    # internal node ratio for positive and negative bags
    pos_ratio = 0.8
    neg_ratio = 0.2

    pos_count = 0
    neg_count = 0

    for i in range(m):  # m = number of bags
        # randomly decide if the bag is positive or negative
        make_positive = random.random() < 0.5  # 50% chance

        if make_positive:
            p = math.ceil(n * pos_ratio)  # number of positive nodes
        else:
            p = math.ceil(n * neg_ratio)  # number of positive nodes for negative bag

        q = n - p  # remaining nodes are negative

        # select nodes
        selected_node = random.sample(label_node[chosen_label], p) + \
                        random.sample(label_non_node[chosen_label], q)

        # shuffle nodes inside the bag
        random.shuffle(selected_node)

        # assign bag label based on intended positive/negative
        positive = 1 if make_positive else 0

        bag_node[tuple(selected_node)] = positive

        # update counters
        if positive == 1:
            pos_count += 1
        else:
            neg_count += 1

    print(f"Mode 3: Number of positive bags = {pos_count}, Number of negative bags = {neg_count}")

    node_train = get_flattened_nodes(bag_node, chosen_label, labels, 0, 300)
    node_eva = get_flattened_nodes(bag_node, chosen_label, labels, 300, 500)

    b = len(node_train)
    c = len(node_eva)
    d = int(0.5 * c)
    idx_val = torch.LongTensor(range(d))
    idx_test = torch.LongTensor(range(d, c))
    idx_train = torch.LongTensor(range(b))

    return idx_train, idx_val, idx_test, node_train, node_eva


def run_experiment(dataset, name, device, hidden_units=64, dropout_rate=0.5,
                   lr=0.01, weight_decay=5e-4, max_epochs=500):

    data = dataset[0].to(device)

    features = sp.csr_matrix(dataset.x, dtype=np.float32)

    # Pick a chosen class for binary classification
    chosen_label = 0
    if name.lower() == "arxiv":
        labels = encode_onehot(dataset.y.squeeze().numpy())
    else:
        labels = encode_onehot(dataset.y.numpy())

    labels = torch.LongTensor(np.where(labels)[1])

    num_class = labels.max().item() + 1
    label_node= dict()
    label_non_node = dict()
    for i in range(num_class):
        label_node[i] = []
        label_non_node[i] = []
    for i in range(labels.shape[0]):
        label_id = labels[i].item()
        label_node[label_id].append(i)
    for i in range(num_class):
        for j in range(num_class):
            if j != i:
                label_non_node[i] += label_node[j]

    m = 500  # number of bag (node-level tasks)
    n = 2  # number of instances within a bag (node-level standard MIL)

    # Pick a chosen class and consistent evaluation nodes

    idx_train, idx_val, idx_test, node_train, node_eva = get_eval_nodes(m, n, chosen_label, data.y, label_node, label_non_node, seed=42)

    # Masks (restrict GCN to only those nodes)
    # data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    # data.train_mask[idx_train] = True
    # data.val_mask[idx_val] = True
    # data.test_mask[idx_test] = True

    node_train_ids = torch.tensor([k[0] for k in node_train.keys()], dtype=torch.long, device=device)
    y_train = torch.tensor(list(node_train.values()), dtype=torch.float, device=device)

    # split node_eva into val/test
    node_eva_keys = list(node_eva.keys())
    split = len(node_eva_keys) // 2
    node_val_ids = torch.tensor([k[0] for k in node_eva_keys[:split]], dtype=torch.long, device=device)
    y_val = torch.tensor([node_eva[k] for k in node_eva_keys[:split]], dtype=torch.float, device=device)
    node_test_ids = torch.tensor([k[0] for k in node_eva_keys[split:]], dtype=torch.long, device=device)
    y_test = torch.tensor([node_eva[k] for k in node_eva_keys[split:]], dtype=torch.float, device=device)

    labels_eva = []
    key_list = list(node_eva.keys())
    for i in range(len(node_eva)):
        key = key_list[i]
        labels_eva.append(node_eva[key])
    labels_eva = torch.LongTensor(labels_eva)

    # ---- GCN training same as before ----
    data = data.to(device)
    model = GCN(dataset.num_features, hidden_units, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # full graph forward
        loss = loss_fn(out[node_train_ids], y_train)  # only flattened nodes
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)

        # Convert logits to predicted labels
        y_pred_train = (out[node_train_ids] > 0).long()
        y_pred_val = (out[node_val_ids] > 0).long()
        y_pred_test = (out[node_test_ids] > 0).long()

        # Compute accuracies
        acc_train = (y_pred_train == y_train.long()).float().mean().item()
        acc_val = (y_pred_val == y_val.long()).float().mean().item()
        acc_test = (y_pred_test == y_test.long()).float().mean().item()

        return acc_train, acc_val, acc_test

    # Run training
    start_time = time.time()
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(1, max_epochs + 1):
        loss = train()
        train_acc, val_acc, test_acc = test()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    end_time = time.time()
    duration = end_time - start_time

    return {
        "dataset": name,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "training_time_sec": duration
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = [
        (CitationFull(root='../datasets/citationfull/', name='Cora_ML'), "Cora_ML"),
        (CitationFull(root='../datasets/citationfull/', name='CiteSeer'), "CiteSeer"),
        (Amazon(root='../datasets/amazon/', name='photo'), "Amazon-Photo"),
        (Actor(root='../datasets/actor/'), "Actor"),
    ]

    results = []
    for dataset, name in experiments:
        print(f"Running experiment on {name}...")
        result = run_experiment(dataset, name, device)
        results.append(result)
        print(f"Finished {name}: {result}\n")

    print("===== Final Results =====")
    for res in results:
        print(res)