from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass,field

from decord import VideoReader

from .analyzer.base import FramesAnalysisResult
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

class Video:
    """Data class to video information"""
    frames: List[Frame]
    video_id: str
    video_name: str
    video_path: str
    analysis: Optional[FramesAnalysisResult] = None

    def __post_init__(self):
        assert isinstance(self.video_path, str), "video_path must be a string"

        vr = VideoReader(self.video_path)
        self.video_duration = vr._num_frame / vr.get_avg_fps()
        self.video_fps = vr.get_avg_fps()
        self.video_size = vr.next().shape

