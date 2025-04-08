import numpy as np
from sklearn.cluster import DBSCAN
from transformers import CLIPProcessor, CLIPModel
import torch
import cv2
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler

# CLIP model initialization
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def get_clip_embedding(text):
    """클래스 이름 텍스트 -> CLIP 임베딩 벡터"""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    vec = embeddings.squeeze().cpu().numpy()  # shape: (512,)
    return vec / np.linalg.norm(vec)  # ✅ 정규화 추가

def cluster_by_clip_and_dbscan(filtered_objects, depth_weight=1.0, clip_weight=0.1, eps=1.2, min_samples=2):
    """
    CLIP + DBSCAN 기반 의미-거리 그룹화
    - filtered_objects: 필터링된 객체 리스트
    - depth_weight: depth 가중치 비율
    - eps, min_samples: DBSCAN 파라미터
    """

    """
    CLIP 임베딩 + 위치/깊이 조합한 벡터를 정규화한 뒤 DBSCAN 적용
    """
    # 1. CLIP 임베딩
    for obj in filtered_objects:
        obj["clip_embed"] = get_clip_embedding(obj["class_name"])

    # 2. 벡터 구성: (x, y, depth, clip_vector)
    vectors = []
    for obj in filtered_objects:
        x, y = obj["center"]
        d = obj["depth"] * depth_weight
        clip_vec = obj["clip_embed"] * clip_weight  # clip 영향 확대
        vec = np.concatenate([[x, y, d], clip_vec])  # shape (515,)
        vectors.append(vec)

    vectors = np.array(vectors)

    # ✅ 3. Standardization 적용 (위치/임베딩 scale 통일)
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    # ✅ 4. DBSCAN 적용
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors_scaled)
    labels = clustering.labels_

    for i, obj in enumerate(filtered_objects):
        obj["cluster_id"] = labels[i]

    print(f"[INFO] DBSCAN + CLIP: {len(set(labels)) - (1 if -1 in labels else 0)} clusters formed")
    return filtered_objects

def draw_clustered_objects(image, clustered_objects, save_path="output/clustered_result.jpg"):
    image_vis = image.copy()
    color_map = {}

    def get_color(cluster_id):
        if cluster_id not in color_map:
            color_map[cluster_id] = [random.randint(0, 255) for _ in range(3)]
        return color_map[cluster_id]

    for obj in clustered_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj["class_name"]
        cluster_id = obj["cluster_id"]
        color = get_color(cluster_id)
        label = f"[{cluster_id}] {cls_name}"

        cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_vis)
    print(f"clustering visualization saved: {save_path}")