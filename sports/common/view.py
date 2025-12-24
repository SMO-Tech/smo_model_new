"""
ViewTransformer compatibility wrapper.
Implements homography transformation using OpenCV.
"""
import numpy as np
import cv2


class ViewTransformer:
    """
    Transforms points between two coordinate systems using homography.
    Compatible with sports.common.view.ViewTransformer API.
    """
    
    def __init__(self, source, target):
        """
        Initialize ViewTransformer with source and target points.
        
        Args:
            source: Source coordinate points (N, 2) array
            target: Target coordinate points (N, 2) array
        """
        if source is None or target is None:
            raise ValueError("Source and target points cannot be None")
        
        source = np.array(source, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        
        if len(source) < 4 or len(target) < 4:
            raise ValueError("Need at least 4 points for homography transformation")
        
        if len(source) != len(target):
            raise ValueError("Source and target must have same number of points")
        
        # Compute homography matrix
        self.homography_matrix, mask = cv2.findHomography(
            source, target, 
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if self.homography_matrix is None:
            raise ValueError("Failed to compute homography matrix")
    
    def transform_points(self, points):
        """
        Transform points from source to target coordinate system.
        
        Args:
            points: Points to transform (N, 2) or (N, 3) array
            
        Returns:
            Transformed points (N, 2) array
        """
        points = np.array(points, dtype=np.float32)
        
        # Handle different input shapes
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # If 3D points (x, y, z), use only x, y
        if points.shape[1] >= 3:
            points = points[:, :2]
        
        # Add homogeneous coordinate
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        
        # Transform
        transformed_homogeneous = (self.homography_matrix @ points_homogeneous.T).T
        
        # Convert back from homogeneous coordinates
        transformed = transformed_homogeneous[:, :2] / transformed_homogeneous[:, 2:3]
        
        return transformed

