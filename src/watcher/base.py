from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """Data class to store detection results"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    frame_timestamp: float

@dataclass
class Frame:
    """Data class to store frame results"""
    frame_id: int
    timestamp: float
    image: np.ndarray
    detections: List[Detection]
    parent_video_id: str
    index_in_video: int