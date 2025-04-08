import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools


SIMILARITY_THRESHOLD = 0.85
W_POS = 0.45 # 0.45
W_DEPTH = 0.3 # 0.3
W_CLIP = 0.25


def compute_pairwise_similarity(obj1, obj2):
    """
    두 객체 간 유사도 계산
    - 위치, depth, CLIP embedding 기준
    """
    pos1, pos2 = np.array(obj1['center']), np.array(obj2['center'])
    pos_sim = 1 - np.linalg.norm(pos1 - pos2) / 600

    depth_diff = abs(obj1['depth'] - obj2['depth'])
    depth_sim = 1 - depth_diff / 10

    clip_sim = cosine_similarity(
        obj1['clip_embed'].reshape(1, -1),
        obj2['clip_embed'].reshape(1, -1)
    )[0][0]

    score = W_POS * pos_sim + W_DEPTH * depth_sim + W_CLIP * clip_sim
    return score, (pos_sim, depth_sim, clip_sim)


def assign_instance_ids(clustered_objects):
    instance_id = 0
    assigned = set()

    for cid in set(obj['cluster_id'] for obj in clustered_objects if obj['cluster_id'] != -1):
        cluster_objs = [obj for obj in clustered_objects if obj['cluster_id'] == cid]

        for i, obj in enumerate(cluster_objs):
            if id(obj) in assigned:
                continue

            obj['instance_id'] = instance_id
            assigned.add(id(obj))

            for j in range(i + 1, len(cluster_objs)):
                other = cluster_objs[j]
                if id(other) in assigned:
                    continue

                sim, _ = compute_pairwise_similarity(obj, other)

                if sim >= SIMILARITY_THRESHOLD:
                    other['instance_id'] = instance_id
                    assigned.add(id(other))

            instance_id += 1

    for obj in clustered_objects:
        if 'instance_id' not in obj:
            obj['instance_id'] = instance_id
            instance_id += 1

    print(f"Instance grouping complete: 총 {instance_id}개의 instance 생성됨")
    return clustered_objects
