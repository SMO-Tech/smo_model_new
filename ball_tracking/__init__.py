"""
Ball Tracking Module

Provides TrackNet-based ball detection as an alternative to YOLO.
"""

from .tracknet_detector import TrackNetDetector, create_tracknet_detector

__all__ = ['TrackNetDetector', 'create_tracknet_detector']
