import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.datasets import (Planetoid, Reddit, Flickr, FacebookPagePage, Actor, LastFMAsia, DeezerEurope,
                                      Amazon, Yelp)
from torch_geometric.utils import to_dense_adj
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

def get_training_data(dataset_choice):

    if dataset_choice == "Cora" or dataset_choice == "Citeseer" or dataset_choice == "PubMed":
        data = load_planetoid_dataset(dataset_choice)
    elif dataset_choice == "Reddit":
        data = load_reddit_dataset()
    elif dataset_choice == "Arxiv":
        data = load_ogbn_dataset(dataset_choice.lower())
    elif dataset_choice == "Products":
        data = load_ogbn_dataset(dataset_choice.lower())
    elif dataset_choice == "Mag":
        data = load_ogbn_dataset(dataset_choice.lower())
    else:
        print("Invalid dataset")
        exit()

    return data

def load_planetoid_dataset(name):
    # Load dataset
    dataset = Planetoid(root=f"../datasets/Planetoid/{name}", name=name)
    data = dataset[0]  # Planetoid datasets contain only one graph



    # Extract features, labels, and masks
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    edge_index = data.edge_index  # Ensure this is in the correct shape

    num_nodes = features.shape[0]

    edges = data.edge_index.numpy().T
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Create the final Data object
    data_obj = Data(
        x=features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        adjacency=adjacency,  # Extra: adjacency matrix
        num_features=features.shape[1],
        num_classes=labels.max().item() + 1,
        edge_index=edge_index
    )

    return data_obj

def load_reddit_dataset(root='../datasets/reddit'):
    # Load Reddit dataset
    dataset = Reddit(root=root)
    data = dataset[0]  # Reddit has a single graph

    # Features, labels, masks
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    edge_index = data.edge_index

    # Adjacency matrix (dense)
    adjacency = None

    # Create custom Data object
    data_obj = Data(
        x=features,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        adjacency=adjacency,  # Extra field
        num_features=features.shape[1],
        num_classes=int(labels.max().item() + 1),
        edge_index=edge_index
    )

    return data_obj

def load_ogbn_dataset(dataset_n):

    dataset_name = f'ogbn-{dataset_n}'
    dataset = PygNodePropPredDataset(name=dataset_name, root='../datasets/')
    data = dataset[0]  # For OGBN-MAG, this is a HeteroData object
    split_idx = dataset.get_idx_split()

    if dataset_n.lower() == 'mag':
        # 1) Features & labels
        features = data.x_dict['paper']  # shape [num_papers, num_features]
        labels = data.y_dict['paper'].view(-1)  # shape [num_papers]

        # 2) Number of paper nodes
        num_nodes = features.size(0)  # or features.shape[0]

        # 3) Citation edges among papers
        edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]

        # 4) Train/val/test splits for "paper" nodes
        train_index = split_idx['train']['paper']
        val_index = split_idx['valid']['paper']
        test_index = split_idx['test']['paper']
    else:
        # --- Homogeneous access for Arxiv/Products ---
        features = data.x
        labels = data.y.squeeze()
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        train_index = split_idx['train']
        val_index = split_idx['valid']
        test_index = split_idx['test']

    # Convert edge_index to a SciPy sparse adjacency matrix:
    adjacency = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    data = Data(x=features, y=labels, train_mask=train_index, val_mask=val_index, test_mask=test_index,
                adjacency=adjacency, num_features=dataset.num_features, num_classes=dataset.num_classes,
                edge_index=edge_index)

    return data