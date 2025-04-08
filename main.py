from scene_analyze import SceneAnalyze
from grouping import cluster_by_clip_and_dbscan, draw_clustered_objects
from instance_grouping import assign_instance_ids
import os
from datetime import datetime
from summary_utils import save_summary_txt


def make_output_folder(base_dir="output"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, now)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    output_dir = make_output_folder()

    # 1. Depth Estimation + Object Detection + Filtering
    analyze = SceneAnalyze("sample/example11.jpg", output_dir=output_dir)
    analyze.run()


    # 2. CLIP + DBSCAN Grouping
    grouped_objects = cluster_by_clip_and_dbscan(analyze.filtered_objects, analyze.image)
    draw_clustered_objects(analyze.image, grouped_objects,
                       save_path=os.path.join(output_dir, "clustered_result.jpg"))

    # 3. Instance Grouping
    grouped_with_instance = assign_instance_ids(grouped_objects)

    draw_clustered_objects(analyze.image, grouped_with_instance,
                       save_path=os.path.join(output_dir, "instance_result.jpg"),
                       use_instance_color=True)

    # Summary text file 저장
    summary_path = os.path.join(output_dir, "summary.txt")
    save_summary_txt(grouped_objects, grouped_with_instance, save_path=summary_path)
