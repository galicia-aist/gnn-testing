# 🧠 GNN Experiment Playground

This project serves as a flexible **sandbox environment** for testing multiple **Graph Neural Networks (GNNs)** across **various architectures** and **graph datasets**.  
The main goal is to easily switch between models, datasets, and configurations — making it simple to compare performance, debug behavior, and extend the framework with new ideas.

---

## 🚀 Overview

The project’s main entry point is `main.py`, which accepts several command-line arguments to configure:

- The **model type** (e.g., GCN, SGC)
- The **dataset** to load (Cora, CiteSeer, PubMed, etc.)
- The **training configuration** (epochs, learning rate, dropout, etc.)
- The **execution setup** (CUDA, distributed training)
- **Logging** and reproducibility controls
- Optional **extra hyperparameters** (`gamma`, `rho`)

All of these are managed through a well-structured `argparse` interface.

---

## ⚙️ Argument Reference

| **Category** | **Argument** | **Type** | **Choices / Default** | **Description** |
|---------------|--------------|-----------|------------------------|-----------------|
| 🧩 **Core setup** | `--model` | `str` | `['GCN', 'SGC']` | Selects which GNN architecture to run. |
| | `--dataset` | `str` | `['Cora', 'CiteSeer', 'PubMed', 'CoraFull', 'CiteseerFull', 'PubMedFull', 'Reddit', 'Products', 'Arxiv', 'Mag']` | Chooses the dataset for training and evaluation. |
| 🏋️ **Training configuration** | `--epochs` | `int` | `500` | Number of training epochs. |
| | `--lr` | `float` | `0.01` | Initial learning rate. |
| | `--weight_decay` | `float` | `5e-4` | L2 regularization strength (weight decay). |
| | `--hidden` | `int` | `16` | Number of hidden units per GNN layer. |
| | `--dropout` | `float` | `0.5` | Dropout rate (1 - keep probability). |
| | `--bag_ratio` | `float` | `0.8` | Ratio controlling positive/negative label sampling in bag-based learning. |
| ⚙️ **Execution setup** | `--cuda` | `flag` | *default:* `False` | Enables GPU (CUDA) training. |
| | `--ddp` | `flag` | *default:* `False` | Enables Distributed Data Parallel (DDP) training. |
| | `--fastmode` | `flag` | *default:* `False` | Skips validation during training for faster runs. |
| 🧮 **Reproducibility & logging** | `--seed` | `int` | `42` | Random seed for reproducibility. |
| | `--log_level` | `str` | `['debug', 'info', 'warning', 'error', 'critical']`, *default:* `info` | Sets the verbosity of logs. |
| | `--log_path` | `str` | `local` | Directory or system path where logs will be saved. |
| ⚗️ **Extra hyperparameters** | `--gamma` | `float` | `0.5` | Custom gamma coefficient for experimental tuning. |
| | `--rho` | `float` | `1.0` | Custom rho coefficient for experimental tuning. |

---

## 🧩 Example Command

Run the project with a specific configuration:

```bash
python main.py \
    --model GCN \
    --dataset Cora \
    --epochs 300 \
    --lr 0.005 \
    --hidden 64 \
    --dropout 0.6 \
    --cuda \
    --log_level debug

```

## 📊 Dataset Overview

Below is a summary of the available graph datasets, including their key statistics —
number of nodes, edges, input features, and classes — grouped by their source (Planetoid, CitationFull, or OGB).

| Group | Dataset | # Nodes | # Edges | # Features | # Classes | Info Link |
|:------|:---------|--------:|--------:|------------:|-----------:|:-----------|
| **Planetoid** | **Cora** | 2,708 | 5,429 | 1,433 | 7 | [PyG Planetoid Datasets](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) |
|  | **CiteSeer** | 3,327 | 4,732 | 3,703 | 6 | ↑ Same as above |
|  | **PubMed** | 19,717 | 44,338 | 500 | 3 | ↑ Same as above |
| **CitationFull** | **CoraFull** | 19,793 | 126,842 | 8,710 | 70 | [PyG CitationFull Datasets](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CitationFull.html#torch_geometric.datasets.CitationFull) |
|  | **Cora_ML** | 2,995 | 16,316 | 2,879 | 7 | ↑ Same as above |
|  | **CiteSeerFull** | 4,230 | 10,674 | 602 | 6 | ↑ Same as above |
|  | **DBLP** | 17,716 | 105,734 | 1,639 | 4 | ↑ Same as above |
|  | **PubMedFull** | 19,717 | 88,648 | 500 | 3 | ↑ Same as above |
| **OGB** | **Reddit** | 232,965 | 11,606,919 | 602 | 41 | [PyG Reddit Dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit.html) |
|  | **Products** | 2,449,029 | 61,859,140 | 100 | 47 | [OGB NodeProp – ogbn-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) |
|  | **Arxiv** | 169,343 | 1,166,243 | 128 | 40 | [OGB NodeProp – ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) |
|  | **Mag** | 1,939,743 (papers only) | 21,111,007 (relations) | 128 | 349 | [OGB NodeProp – ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) |
