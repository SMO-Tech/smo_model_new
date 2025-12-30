import cv2
import numpy as np
from typing import Optional, Tuple, List

from keypoint_detection.keypoint_constants import (
    FIELD_CORNERS,
    FIFA_FIELD_LENGTH,
    FIFA_FIELD_WIDTH,
    CONFIDENCE_THRESHOLD,
)


class PitchCalibration:
    """Compute a one-time homography between frame pixels and pitch meters."""

    def __init__(self, min_confidence: float = CONFIDENCE_THRESHOLD, inlier_ratio: float = 0.6):
        self.min_confidence = min_confidence
        self.inlier_ratio = inlier_ratio
        self.image_points: Optional[np.ndarray] = None
        self.pitch_points: Optional[np.ndarray] = None
        self.homography: Optional[np.ndarray] = None
        self.enabled = False

        corners = FIELD_CORNERS
        length = FIFA_FIELD_LENGTH
        width = FIFA_FIELD_WIDTH
        self.pitch_map = {
            corners['top_left']: (0.0, 0.0),
            corners['top_right']: (length, 0.0),
            corners['bottom_right']: (length, width),
            corners['bottom_left']: (0.0, width),
        }

    def try_calibrate(self, detected_keypoints: Optional[np.ndarray]) -> bool:
        """
        Compute homography once using reliable corner keypoints.

        Returns True if calibration succeeded (or already active).
        """
        if self.enabled:
            return True
        if detected_keypoints is None or detected_keypoints.shape[0] == 0:
            return False

        keypoints = detected_keypoints[0]
        image_pts: List[Tuple[float, float]] = []
        pitch_pts: List[Tuple[float, float]] = []

        for idx, pitch_coord in self.pitch_map.items():
            if idx >= keypoints.shape[0]:
                continue
            x, y, conf = keypoints[idx]
            if conf < self.min_confidence or x <= 0 or y <= 0:
                continue
            image_pts.append((x, y))
            pitch_pts.append(pitch_coord)

        if len(image_pts) < 4:
            return False

        H, status = cv2.findHomography(
            np.array(image_pts, dtype=np.float32),
            np.array(pitch_pts, dtype=np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if H is None or status is None:
            return False

        inlier_ratio = status.sum() / len(status)
        if inlier_ratio < self.inlier_ratio:
            return False

        self.image_points = np.array(image_pts)
        self.pitch_points = np.array(pitch_pts)
        self.homography = H
        self.enabled = True
        return True

    def pixel_to_pitch(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to pitch meters."""
        if not self.enabled or self.homography is None:
            return None

        pts = np.array([[[x, y]]], dtype=np.float32)
        try:
            mapped = cv2.perspectiveTransform(pts, self.homography)
        except cv2.error:
            return None
        xp, yp = mapped[0][0]
        return float(xp), float(yp)

