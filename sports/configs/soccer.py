"""
SoccerPitchConfiguration compatibility wrapper.
Provides standard soccer field keypoint coordinates.
"""
import numpy as np


class SoccerPitchConfiguration:
    """
    Configuration for standard soccer field keypoints.
    Provides vertices for FIFA-standard field (105m x 68m).
    Coordinates are in a standard pitch coordinate system.
    """
    
    def __init__(self):
        """Initialize with FIFA standard field dimensions."""
        # FIFA standard field: 105m x 68m
        # Using a coordinate system where field is 12000 x 8000 units
        # (scaled for better precision)
        
        field_length = 12000  # 105m scaled
        field_width = 8000    # 68m scaled
        
        # Field corners
        self.vertices = np.array([
            [0, 0],                    # 0: Top-left corner
            [0, field_width],          # 1: Bottom-left corner  
            [field_length, 0],         # 2: Top-right corner
            [field_length, field_width], # 3: Bottom-right corner
            
            # Left penalty area
            [0, field_width/2 - 2015],      # 4: Left penalty top-left
            [1650, field_width/2 - 2015],   # 5: Left penalty top-right
            [0, field_width/2 + 2015],      # 6: Left penalty bottom-left
            [1650, field_width/2 + 2015],   # 7: Left penalty bottom-right
            
            # Left goal area
            [0, field_width/2 - 915],       # 8: Left goal top-left
            [550, field_width/2 - 915],    # 9: Left goal top-right
            [0, field_width/2 + 915],       # 10: Left goal bottom-left
            [550, field_width/2 + 915],     # 11: Left goal bottom-right
            
            # Right penalty area
            [field_length - 1650, field_width/2 - 2015],  # 12: Right penalty top-left
            [field_length, field_width/2 - 2015],          # 13: Right penalty top-right
            [field_length - 1650, field_width/2 + 2015], # 14: Right penalty bottom-left
            [field_length, field_width/2 + 2015],          # 15: Right penalty bottom-right
            
            # Right goal area
            [field_length - 550, field_width/2 - 915],    # 16: Right goal top-left
            [field_length, field_width/2 - 915],          # 17: Right goal top-right
            [field_length - 550, field_width/2 + 915],   # 18: Right goal bottom-left
            [field_length, field_width/2 + 915],          # 19: Right goal bottom-right
            
            # Center line
            [field_length/2, 0],           # 20: Center line top
            [field_length/2, field_width], # 21: Center line bottom
            
            # Center circle points
            [field_length/2 - 915, field_width/2],  # 22: Center circle left
            [field_length/2 + 915, field_width/2],  # 23: Center circle right
            [field_length/2, field_width/2 - 915],  # 24: Center circle top
            [field_length/2, field_width/2 + 915],   # 25: Center circle bottom
            
            # Additional reference points
            [field_length/4, field_width/2],      # 26: Left quarter
            [3*field_length/4, field_width/2],   # 27: Right quarter
            [field_length/2, field_width/4],      # 28: Top quarter
            [field_length/2, 3*field_width/4],   # 29: Bottom quarter
            
            # Penalty arcs (approximate)
            [1650, field_width/2],                 # 30: Left penalty arc center
            [field_length - 1650, field_width/2],  # 31: Right penalty arc center
        ], dtype=np.float32)

