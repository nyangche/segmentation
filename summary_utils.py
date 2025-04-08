from instance_grouping import compute_pairwise_similarity
from instance_grouping import (
    compute_pairwise_similarity,
    SIMILARITY_THRESHOLD,
    W_POS,
    W_DEPTH,
    W_CLIP
)


def save_summary_txt(grouped_objects, grouped_with_instance, save_path, similarity_threshold=0.7):
    with open(save_path, "w") as f:
        f.write("# ===== Experiment Config =====\n")
        f.write(f"similarity_threshold = {SIMILARITY_THRESHOLD}\n")
        f.write(f"w_pos = {W_POS}\n")
        f.write(f"w_depth = {W_DEPTH}\n")
        f.write(f"w_clip = {W_CLIP}\n\n")

        f.write("[Cluster Result]\n")
        for obj in grouped_objects:
            cid = obj["cluster_id"]
            name = obj["class_name"]
            center = obj["center"]
            bbox = obj["bbox"]
            depth = obj["depth"]
            f.write(f"[Cluster {cid}] {name} at {center}, Depth={depth:.2f}, BBox={bbox}\n")
        
        f.write("\n[Instance Grouping Result]\n")
        for obj in grouped_with_instance:
            iid = obj["instance_id"]
            name = obj["class_name"]
            center = obj["center"]
            bbox = obj["bbox"]
            depth = obj["depth"]
            f.write(f"[Instance {iid}] {name} at {center}, Depth={depth:.2f}, BBox={bbox}\n")

        f.write("------------------\n")
        f.write("\n[Pairwise Similarity (above threshold)]\n")
        for i in range(len(grouped_with_instance)):
            for j in range(i + 1, len(grouped_with_instance)):
                obj1 = grouped_with_instance[i]
                obj2 = grouped_with_instance[j]

                if obj1["cluster_id"] != obj2["cluster_id"]:
                    continue

                sim, (pos_sim, depth_sim, clip_sim) = compute_pairwise_similarity(obj1, obj2)

                if sim >= similarity_threshold:
                    f.write(f"\n({obj1['class_name']} #{obj1['instance_id']}) & "
                            f"({obj2['class_name']} #{obj2['instance_id']})\n")
                    f.write(f"  - total: {sim:.3f}, pos: {pos_sim:.3f}, depth: {depth_sim:.3f}, clip: {clip_sim:.3f}\n")

    print(f"summary.txt 저장 완료: {save_path}")
