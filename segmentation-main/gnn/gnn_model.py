# 간단한 gcn 모델 정의 (PyG)

# gnn/gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ObjectGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super(ObjectGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # logits