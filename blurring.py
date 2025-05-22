def draw_focus_overlay_by_class(original_image_path, segmentation_image_path, grouped_objects, target_class, save_path):
    import cv2
    import numpy as np
    import os

    image = cv2.imread(original_image_path)
    seg_map = cv2.imread(segmentation_image_path)

    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    target_ids = {
        obj["instance_id"] + 1
        for obj in grouped_objects if obj["class_name"].lower() == target_class.lower()
    }

    np.random.seed(42)
    instance_ids = {obj["instance_id"] for obj in grouped_objects}
    color_map = {
        inst_id + 1: np.random.randint(0, 255, size=3).tolist()
        for inst_id in instance_ids
    }
    inverse_map = {tuple(v): k for k, v in color_map.items()}

    mask = np.zeros((h, w), dtype=np.uint8)
    for color_rgb, inst_id in inverse_map.items():
        binary_mask = np.all(seg_map == color_rgb, axis=-1)
        if inst_id in target_ids:
            mask[binary_mask] = 1

    result = np.where(mask[:, :, None] == 1, image, blurred)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, result)
    print(f"Focused overlay saved to: {save_path}")
