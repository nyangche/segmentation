import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch
import cv2
import os
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize


# CLIP model initialization
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def get_clip_embedding(text):
    """클래스 이름 텍스트 -> CLIP 임베딩 벡터"""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    return embeddings.squeeze().cpu().numpy()  # shape: (512,)


def cluster_by_clip_and_dbscan(filtered_objects, image, depth_weight=3.0, eps=3.5, min_samples=2):

    # CLIP + DBSCAN 기반 의미-거리 그룹화
    # - filtered_objects: 필터링된 객체 리스트
    # - depth_weight: depth 가중치 비율
    # - eps, min_samples: DBSCAN 파라미터

    # 1. get CLIP embedding
    for obj in filtered_objects:
        obj["clip_embed"] = get_clip_embedding(obj["class_name"])

    # 2. 벡터 구성: (center_x, center_y, depth, clip_embed)
    vectors = []
    for obj in filtered_objects:
        # x, y = obj["center"]

        h, w = image.shape[:2]  
        x, y = obj["center"]
        x = x / w
        y = y / h

        d = obj["depth"] * depth_weight

        # clip_vec = obj["clip_embed"]
        # vec = np.concatenate([[x, y, d], clip_vec])

        clip_vec = obj["clip_embed"]

        clip_vec = clip_vec / np.linalg.norm(clip_vec) 
        clip_vec = clip_vec * 8.0  

        vec = np.concatenate([[x, y, d], clip_vec])
        vectors.append(vec)

    vectors = np.array(vectors)  # shape: (N, 3+512)

    # 3. DBSCAN clustering
    # clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)

    vectors = normalize(vectors)
    clustering = DBSCAN(eps=0.15, min_samples=min_samples, metric='cosine').fit(vectors)

    labels = clustering.labels_

    for i, obj in enumerate(filtered_objects):
        obj["cluster_id"] = labels[i]

    print(f"DBSCAN + CLIP clustering done: 총 {len(set(labels)) - (1 if -1 in labels else 0)} groups")
    return filtered_objects



def draw_clustered_objects(image, clustered_objects, save_path="output/clustered_result.jpg", use_instance_color=True):
    image_vis = image.copy()
    color_map = {}

    def get_color(key):
        if key not in color_map:
            color_map[key] = [random.randint(0, 255) for _ in range(3)]
        return color_map[key]

    for obj in clustered_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj["class_name"]
        
        color_key = obj["instance_id"] if use_instance_color and "instance_id" in obj else obj["cluster_id"]
        color = get_color(color_key)
        
        label = f"[{color_key}] {cls_name}"

        cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_vis)
    print(f"clustering visualization saved: {save_path}")



def draw_focus_instance_blur_rest(image, clustered_objects, focus_class_name, save_path="output/focus_result.jpg"):
    image_vis = image.copy()
    blurred = cv2.GaussianBlur(image_vis, (55, 55), 0)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    color = [random.randint(0, 255) for _ in range(3)]

    for obj in clustered_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj["class_name"]

        if cls_name == focus_class_name:
            
            cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_vis, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            mask[y1:y2, x1:x2] = 1

    # blurring
    result = np.where(mask[:, :, None] == 1, image_vis, blurred)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, result)
    print(f"Focused result saved to {save_path}")
