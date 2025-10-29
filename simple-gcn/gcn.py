import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, data, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, hidden)
        self.conv2 = GCNConv(hidden, data.num_classes)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)
