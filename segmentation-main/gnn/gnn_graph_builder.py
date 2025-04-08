# filtered_objects → PyG graph로 바꾸는 함수

# gnn/gnn_graph_builder.py
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_object_graph(filtered_objects):
    node_features = []
    edge_index = []
    edge_attr = []

    # 1. 노드 feature 만들기
    for obj in filtered_objects:
        x, y = obj["center"]
        d = obj["depth"]
        clip_vec = obj["clip_embed"]
        clip_vec = clip_vec / np.linalg.norm(clip_vec) * 0.05
        feature = np.concatenate([[x, y, d], clip_vec])
        node_features.append(feature)

    node_features = torch.tensor(node_features, dtype=torch.float)  # shape: [num_nodes, feat_dim]

    # 2. edge 만들기 (모든 쌍 간 연결 or 조건 제한)
    num_nodes = len(filtered_objects)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            xi, yi = filtered_objects[i]["center"]
            xj, yj = filtered_objects[j]["center"]
            dist = np.linalg.norm(np.array([xi, yi]) - np.array([xj, yj]))

            if dist < 150:  # 임계 거리 내만 연결 (조정 가능)
                edge_index.append([i, j])
                edge_attr.append([dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data