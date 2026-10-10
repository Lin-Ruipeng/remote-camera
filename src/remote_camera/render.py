"""Annotation uses the exact frame associated with the detection result."""

import cv2
import numpy as np

from .pose import SKELETON, PoseResult

COLORS = ((99, 218, 84), (252, 174, 72), (242, 119, 203), (84, 208, 249))


def annotate(result: PoseResult, keypoint_confidence: float = 0.5) -> np.ndarray:
    image = result.frame.image_bgr.copy()
    for index, (box, points) in enumerate(zip(result.boxes, result.keypoints, strict=True)):
        color = COLORS[index % len(COLORS)]
        x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"Person {index + 1}  {box[4]:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        visible = points[:, 2] >= keypoint_confidence
        coords = np.rint(points[:, :2]).astype(int)
        for first, second in SKELETON:
            if visible[first] and visible[second]:
                cv2.line(image, tuple(coords[first]), tuple(coords[second]), color, 2, cv2.LINE_AA)
        for point, show in zip(coords, visible, strict=True):
            if show:
                cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)
    return image
