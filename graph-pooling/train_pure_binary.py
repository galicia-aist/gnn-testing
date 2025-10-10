import random

import torch
import torch.nn.functional as F
import time
from torch_geometric.datasets import CitationFull, Amazon, Actor
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split


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


def run_experiment(dataset, name, device, hidden_units=64, dropout_rate=0.5,
                   lr=0.01, weight_decay=5e-4, max_epochs=500):

    data = dataset[0]

    # Binary labels: class 0 vs rest
    binary_labels = (data.y == 0).long()
    data.y = binary_labels

    # Split into 60/20/20
    num_nodes = data.num_nodes
    idx = torch.arange(num_nodes)

    train_idx, temp_idx = train_test_split(idx, test_size=0.4, stratify=data.y, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=data.y[temp_idx], random_state=42)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    data = data.to(device)
    model = GCN(dataset.num_features, hidden_units, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = (out > 0).long()

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            correct = (pred[mask] == data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())
            accs.append(acc)
        return accs

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
        (CitationFull(root='./citationfull/', name='Cora_ML'), "Cora_ML"),
        (CitationFull(root='./citationfull/', name='CiteSeer'), "CiteSeer"),
        (Amazon(root='./amazon/', name='photo'), "Amazon-Photo"),
        (Actor(root='./actor/'), "Actor"),
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