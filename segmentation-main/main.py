from scene_analyze import SceneAnalyze
from grouping import draw_clustered_objects, cluster_by_clip_and_dbscan


if __name__ == "__main__":
    # 1. Depth Estimation + Object Detection + Filtering
    analyze = SceneAnalyze("../COCO/images/val2017/000000013659.jpg")
    #analyze = SceneAnalyze("../COCO/images/val2017/000000054593.jpg")
    #analyze = SceneAnalyze("Depth-Anything-V2/assets/examples/demo08.jpg") 
    # 'sample/sample.jpg'->'Depth-Anything-V2/assets/examples/demo05.jpg'
    analyze.run()

    print("[DEBUG] 필터링된 객체 수:", len(analyze.filtered_objects))

# #     # 2. CLIP + DBSCAN Grouping
    
#     # grouped_objects = cluster_by_clip_and_dbscan(analyze.filtered_objects)
#     # 1차 필터링 + 그룹화
    grouped_objects = cluster_by_clip_and_dbscan(
    analyze.filtered_objects, 
    depth_weight=10,  # ⬅ depth 비중 강화
    clip_weight=2.0,     # 의미 유사도 반영
    eps=1.2,             # 표준화 후 거리 기준 (0.8~1.5 권장)
    min_samples=1       # ⬅ 최소 샘플 수 낮춤
    
)
    # 3. 시각화
    draw_clustered_objects(analyze.image, grouped_objects)


    # console 출력
    for obj in grouped_objects:
        cid = obj["cluster_id"]
        name = obj["class_name"]
        depth = obj["depth"]
        center = obj["center"]
        print(f"[Cluster {cid}] {name} at {center}, depth={depth:.2f}")
