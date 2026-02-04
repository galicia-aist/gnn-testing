from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, single_layer=False):
        super().__init__()
        self.single_layer = single_layer
        self.dropout = dropout
        out_channels = 1
        mid_channels = out_channels if single_layer else hidden_channels
        self.conv1 = GCNConv(in_channels, mid_channels)
        self.conv2 = None if single_layer else GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if not self.single_layer:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        return x.squeeze(-1)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, heads=1, single_layer=False):
        super().__init__()
        self.single_layer = single_layer
        self.dropout = dropout
        out_channels = 1
        mid_channels = out_channels if single_layer else hidden_channels * heads
        self.conv1 = GATConv(
            in_channels,
            mid_channels if single_layer else hidden_channels,
            heads=1 if single_layer else heads,
            concat=not single_layer
        )
        self.conv2 = None if single_layer else GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if not self.single_layer:
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        return x.squeeze(-1)




class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, single_layer=False):
        super().__init__()
        self.single_layer = single_layer
        self.dropout = dropout
        out_channels = 1
        mid_channels = out_channels if single_layer else hidden_channels
        self.conv1 = SAGEConv(in_channels, mid_channels)
        self.conv2 = None if single_layer else SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if not self.single_layer:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        return x.squeeze(-1)
