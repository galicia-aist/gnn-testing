# 🧠 GNN Testing Suite

This repository is a **collection of experimental Graph Neural Network (GNN) projects** designed to **test and reproduce existing research papers** under different configurations, datasets, and architectural variations.  
The goal is to create a flexible environment for exploring how GNN methods behave across multiple setups — from simple baselines to complex pooling mechanisms.

---

## 🔬 Project Overview

| Subproject | Description | Paper / Concept | Link |
|-------------|--------------|-----------------|------|
| **Graph Pooling (Instance-Based GNN)** | Implements **Instance-Based Graph Pooling**, a novel approach for **weakly supervised node learning**. This method focuses on **instance-level pooling** rather than embedding-based pooling, improving performance in **Multiple Instance Learning (MIL)** graph settings. | [Instance-Based Graph Pooling for Weakly Supervised Node Learning](https://arxiv.org/abs/2406.09187) | [🧩 View Project](https://github.com/galicia-aist/gnn-testing/tree/main/graph-pooling) |
| **Simple GCN Benchmark** | A minimal, extensible framework to test **standard GNN models (e.g., GCN, SGC)** on various datasets. This serves as a baseline environment to explore **different training settings, dataset scales, and distributed setups (DDP)**. | Standard GCN / SGC implementations and experimental training variations | [⚙️ View Project](https://github.com/galicia-aist/gnn-testing/tree/main/simple-gcn) |

---

## 🧩 Purpose

The **GNN Testing Suite** is intended for:
- Reproducing **key GNN research results** in a modular setup.  
- Evaluating **variations of architectures and hyperparameters** under unified scripts.  
- Comparing model performance across multiple datasets (Cora, CiteSeer, PubMed, OGB datasets, etc.).  
- Providing a **clean codebase** for quick testing of new ideas on existing graph models.

---

## 🧱 Structure

```
gnn-testing/
│
├── graph-pooling/      # Instance-Based Pooling implementation
├── simple-gcn/         # Baseline GCN / SGC experiments
└── shared/             # Common utilities and dataset loaders (optional)
```

Each subproject includes its own `README.md` with detailed usage instructions, configurations, and dataset details.

---


## 🧩 Example Command

Run the Simple GCN experiment:

```bash
python main.py --model GCN --dataset Cora --epochs 500 --lr 0.01
```

---

## 🧑‍💻 Author

Developed by **Gustavo Galicia**

---

## 🧾 License

This project is licensed under the MIT License — see the LICENSE file for details.
