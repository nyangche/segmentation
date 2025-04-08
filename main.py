from scene_analyze import SceneAnalyze
from grouping.grouping import cluster_by_clip_and_dbscan, draw_clustered_objects
from grouping.gnn import gnn_for_grouping

if __name__ == "__main__":
    # 1. Depth Estimation + Object Detection + Filtering
    analyze = SceneAnalyze(r'C:\IoT_project\segmentation_iot-1\dataset\val2017 (1)\val2017\000000577932.jpg') 
    # 'sample/sample.jpg'->'C:\IoT_project\segmentation_iot-1\dataset\val2017 (1)\val2017\000000577932.jpg'
    analyze.run()

    print("[DEBUG] 필터링된 객체 수:", len(analyze.filtered_objects))

    # 2. CLIP + DBSCAN Grouping
    grouped_objects = cluster_by_clip_and_dbscan(
        analyze.filtered_objects,
        analyze.image,
        depth_weight=1.0,
        eps=0.3,
        min_samples=2
    )
    #grouped_objects = gnn_for_grouping(grouped_objects)
    draw_clustered_objects(analyze.image, grouped_objects)


    for obj in grouped_objects:
        cid = obj["cluster_id"]
        name = obj["class_name"]
        depth = obj["depth"]
        center = obj["center"]
        print(f"[Cluster {cid}] {name} at {center}, depth={depth:.2f}")
