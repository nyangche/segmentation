# gnn/gnn_infer.py
import torch
from gnn.gnn_model import ObjectGCN
from gnn.gnn_graph_builder import build_object_graph

def run_gnn_clustering(filtered_objects, model_ckpt=None):
    data = build_object_graph(filtered_objects)
    input_dim = data.x.shape[1]

    model = ObjectGCN(input_dim)
    if model_ckpt:
        model.load_state_dict(torch.load(model_ckpt))

    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=1).cpu().numpy()

    for i, obj in enumerate(filtered_objects):
        obj["cluster_id"] = int(pred[i])
    return filtered_objects