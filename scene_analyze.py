# scene_analyzer.py

import cv2
from depth.depth import load_depth_model, infer_depth, save_depth_map
from object.object import load_yolo, detect_objects, visualize_detections, filter_detections, draw_filtered_objects

"""
Depth Estimation + Object Detection + Filtering까지 실행
- Depth: DepthAnythingV2
- Object Detection: YOLOv8s
- filtering
"""

class SceneAnalyze:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.img_path = image_path
        self.depth_model = load_depth_model()
        self.yolo_model = load_yolo()
        self.depth_map = None
        self.yolo_results = None
        self.filtered_objects = []

    def run_depth_estimation(self):
        self.depth_map = infer_depth(self.depth_model, self.image)
        save_depth_map(self.depth_map)

    def run_object_detection(self):
        self.yolo_results = detect_objects(self.yolo_model, self.image)
        visualize_detections(self.image, self.yolo_results, self.depth_map, self.yolo_model.names)

    def filter_objects(self):
        self.filtered_objects = filter_detections(self.yolo_results, self.depth_map, class_names=self.yolo_model.names)

    def draw_filtered(self):
        draw_filtered_objects(self.image, self.filtered_objects, self.yolo_model.names)

    def run(self):
        print(f"Analyzed image: {self.img_path}")
        self.run_depth_estimation()
        self.run_object_detection()
        self.filter_objects()
        self.draw_filtered()