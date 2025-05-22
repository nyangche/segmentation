import os
from datetime import datetime

def make_output_folder(base_dir="static/output"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, now)
    os.makedirs(path, exist_ok=True)
    return path


def run_pipeline(image_path, prompt=None, base_output_dir="static/output"):
    import time
    from scene_analyze import SceneAnalyze
    from grouping import cluster_by_clip_and_dbscan, draw_clustered_objects
    from instance_grouping import assign_instance_ids
    from segmentation import segment_instance_groups, overlay_instance_segmentation_sam
    from summary_utils import save_summary_txt

    output_dir = make_output_folder()
    log_lines = []

    total_start = time.time()

    # 1. Depth Estimation + Object Detection + Filtering
    t1 = time.time()
    analyze = SceneAnalyze(image_path, output_dir=output_dir)
    analyze.run()
    t2 = time.time()
    log_lines.append(f"[1] Depth + Detection + Filtering: {t2 - t1:.2f} sec")

    # 2. CLIP + DBSCAN Grouping
    t3 = time.time()
    grouped_objects = cluster_by_clip_and_dbscan(analyze.filtered_objects, analyze.image)
    draw_clustered_objects(analyze.image, grouped_objects,
                           save_path=os.path.join(output_dir, "clustered_result.jpg"))
    t4 = time.time()
    log_lines.append(f"[2] CLIP + DBSCAN Grouping: {t4 - t3:.2f} sec")

    # 3. Instance Grouping
    t5 = time.time()
    grouped_with_instance = assign_instance_ids(grouped_objects)
    draw_clustered_objects(analyze.image, grouped_with_instance,
                           save_path=os.path.join(output_dir, "instance_result.jpg"),
                           use_instance_color=True)
    t6 = time.time()
    log_lines.append(f"[3] Instance Grouping: {t6 - t5:.2f} sec")

    # 4. SAM Segmentation + Overlay
    t7 = time.time()
    segmentation_path = os.path.join(output_dir, "sam_seg_result.png")
    instance_mask_array, colored_mask, color_map = segment_instance_groups(analyze.image, grouped_with_instance, segmentation_path)
    overlay_instance_segmentation_sam(analyze.img_path, segmentation_path, os.path.join(output_dir, "overlay.png"))
    t8 = time.time()
    log_lines.append(f"[4] SAM Segmentation + Overlay: {t8 - t7:.2f} sec")

    total_end = time.time()
    log_lines.append(f"Total Time: {total_end - total_start:.2f} sec")

    # 5. Text-based Emphasis (Handled in demo.py)
    t9 = time.time()
    if prompt:
        log_lines.append(f"[Streamlit] 사용자 입력 강조 요청: {prompt}")
    else:
        log_lines.append("[Streamlit] 강조 요청 없음")

    t10 = time.time()
    log_lines.append(f"[5] Focus + Blur (Handled in Streamlit): {t10 - t9:.2f} sec")

    # Summary 저장
    summary_path = os.path.join(output_dir, "summary.txt")
    save_summary_txt(grouped_objects, grouped_with_instance, save_path=summary_path)

    # 시간 정보 저장 추가
    with open(summary_path, "a") as f:
        f.write("\n\n--- Time Log ---\n")
        for line in log_lines:
            f.write(line + "\n")

    return os.path.join(output_dir, "overlay.png"), output_dir, grouped_with_instance, instance_mask_array, colored_mask, color_map



