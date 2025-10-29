import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.datasets import (Planetoid, Reddit, Flickr, FacebookPagePage, Actor, LastFMAsia, DeezerEurope,
                                      Amazon, Yelp)
from torch_geometric.utils import to_dense_adj


def get_training_data(dataset_choice):

    if dataset_choice == "Cora" or dataset_choice == "Citeseer" or dataset_choice == "PubMed":
        data = load_planetoid_dataset(dataset_choice)
    elif dataset_choice == "Reddit":
        data = load_reddit_dataset()
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