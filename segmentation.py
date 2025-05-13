import os
import numpy as np
import torch
import cv2
from PIL import Image
from collections import defaultdict
from segment_anything import sam_model_registry, SamPredictor

# SAM으로로 바꿈
# vit_b, vit_l은 성능 별로임 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
sam.to(device)
predictor = SamPredictor(sam)

def segment_instance_groups(image, grouped_objects, output_path):
    h, w = image.shape[:2]
    mask_total = np.zeros((h, w), dtype=np.uint8)

    predictor.set_image(image)

    instance_groups = defaultdict(list)
    for obj in grouped_objects:
        instance_groups[obj['instance_id']].append(obj)

    for instance_id, objs in instance_groups.items():
        # 객체들 각각의 bbox로 마스크 예측하고 누적
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        for obj in objs:
            x1, y1, x2, y2 = obj["bbox"]
            box = np.array([x1, y1, x2, y2])
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
            mask = masks[0].astype(np.uint8)

            # 마스크 후처리
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            instance_mask = np.logical_or(instance_mask, mask).astype(np.uint8)


        # 전체 마스크에 instance_id로 마킹
        mask_total = np.where((instance_mask > 0) & (mask_total == 0), instance_id + 1, mask_total)

    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(max(instance_groups.keys()) + 2, 3), dtype=np.uint8)
    for i in range(1, len(colors)):
        colored_mask[mask_total == i] = colors[i]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, colored_mask)
    print(f"Segmentation result saved to: {output_path}")


def overlay_instance_segmentation_sam(img_path, segmentation_path, save_path="overlay.png"):
    image = cv2.imread(img_path)
    seg_map = cv2.imread(segmentation_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    result = image.copy()

    dimmed_background = (image * 0.4 + 255 * 0.6).astype(np.uint8)

    foreground_mask = np.zeros((h, w), dtype=np.uint8)

    unique_colors = np.unique(seg_map.reshape(-1, 3), axis=0)
    np.random.seed(42)
    color_map = {
        tuple(color): tuple(np.random.randint(0, 255, 3).tolist())
        for color in unique_colors if not np.all(color == [0, 0, 0])
    }

    for color in unique_colors:
        if np.all(color == [0, 0, 0]):
            continue

        binary_mask = np.all(seg_map == color, axis=-1).astype(np.uint8)
        foreground_mask = np.logical_or(foreground_mask, binary_mask).astype(np.uint8)

        contour_mask = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        border_color = color_map[tuple(color)]
        cv2.drawContours(result, contours, -1, border_color, thickness=2)

    for c in range(3):
        result[:, :, c] = np.where(foreground_mask == 1,
                                   result[:, :, c],
                                   dimmed_background[:, :, c])

    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, result)
    print(f"SAM overlay saved to: {save_path}")