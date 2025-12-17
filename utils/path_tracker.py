"""
Ball Path Tracker - Visualizes ball trajectory over time
Inspired by the AbsolutePath class but optimized for our OpenCV-based system
"""

import cv2
import numpy as np
from collections import deque
from utils import get_center_of_bbox

class BallPathTracker:
    def __init__(self, max_history=60, fade_alpha=True):
        """
        Initialize ball path tracker
        
        Args:
            max_history: Maximum number of points to keep in path
            fade_alpha: Whether to fade older points (alpha blending)
        """
        self.max_history = max_history
        self.fade_alpha = fade_alpha
        self.path_points = deque(maxlen=max_history)  # [(x, y, frame_num), ...]
        self.colors = deque(maxlen=max_history)  # Track colors for each point
        
    def add_point(self, bbox, frame_num, color=(0, 255, 0), is_predicted=False):
        """
        Add a new point to the path
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_num: Current frame number
            color: Color for this point (BGR format)
            is_predicted: Whether this is a predicted position
        """
        center = get_center_of_bbox(bbox)
        self.path_points.append((center[0], center[1], frame_num))
        
        # Use different color for predicted positions
        if is_predicted:
            color = (0, 165, 255)  # Orange for predictions
        
        self.colors.append(color)
    
    def draw_path(self, frame, thickness=3, draw_arrows=True, arrow_frequency=15):
        """
        Draw the ball path on the frame
        
        Args:
            frame: Frame to draw on (numpy array)
            thickness: Line thickness
            draw_arrows: Whether to draw direction arrows
            arrow_frequency: Draw arrow every N points
            
        Returns:
            Frame with path drawn
        """
        if len(self.path_points) < 2:
            return frame
        
        frame = frame.copy()
        points = list(self.path_points)
        colors = list(self.colors)
        
        # Draw path lines with fading alpha
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            
            # Calculate alpha based on position in path
            if self.fade_alpha:
                # Older points are more transparent
                alpha = 0.3 + 0.7 * (i / len(points))
            else:
                alpha = 1.0
            
            # Blend color with alpha
            color = colors[i]
            blended_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, pt1, pt2, blended_color, thickness)
        
        # Draw direction arrows
        if draw_arrows and len(points) >= arrow_frequency:
            for i in range(arrow_frequency, len(points), arrow_frequency):
                if i < len(points):
                    start_idx = max(0, i - 5)  # Look back 5 points
                    start_pt = points[start_idx]
                    end_pt = points[i]
                    
                    self._draw_arrow(frame, 
                                   (int(start_pt[0]), int(start_pt[1])),
                                   (int(end_pt[0]), int(end_pt[1])),
                                   colors[i],
                                   thickness=thickness)
        
        return frame
    
    def _draw_arrow(self, frame, start, end, color, length=15, thickness=3):
        """
        Draw an arrow from start to end point
        
        Args:
            frame: Frame to draw on
            start: Start point (x, y)
            end: End point (x, y)
            color: Arrow color (BGR)
            length: Arrow head length
            thickness: Arrow thickness
        """
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Vector length
        vec_len = np.sqrt(dx*dx + dy*dy)
        
        if vec_len == 0:
            return
        
        # Normalized direction
        ux = dx / vec_len
        uy = dy / vec_len
        
        # Perpendicular vector
        perp_x = -uy
        perp_y = ux
        
        # Arrow head points
        arrow_end = end
        arrow_width = 8
        
        left_x = int(end[0] - length * ux + arrow_width * perp_x)
        left_y = int(end[1] - length * uy + arrow_width * perp_y)
        
        right_x = int(end[0] - length * ux - arrow_width * perp_x)
        right_y = int(end[1] - length * uy - arrow_width * perp_y)
        
        # Draw arrow head
        cv2.line(frame, (left_x, left_y), arrow_end, color, thickness)
        cv2.line(frame, (right_x, right_y), arrow_end, color, thickness)
    
    def clear(self):
        """Clear the path history"""
        self.path_points.clear()
        self.colors.clear()
    
    def get_path_length(self):
        """Get current path length"""
        return len(self.path_points)
    
    def get_recent_points(self, n=10):
        """Get the most recent N points"""
        return list(self.path_points)[-n:]

