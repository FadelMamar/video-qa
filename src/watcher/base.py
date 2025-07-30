from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass,field
from enum import Enum
import uuid
from decord import VideoReader


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
    image: np.ndarray
    timestamp: float = 0.0
    detections: Optional[List[Detection]] = None
    parent_video_id: str = ""

@dataclass
class FramesAnalysisResult:
    """Container for video analysis results."""
    frames_analysis: List[str]
    timestamps: List[float]
    summary: str
    
    def __post_init__(self):
        """Validate the analysis result data."""
        if len(self.frames_analysis) != len(self.timestamps):
            raise ValueError("Number of frame analyses must match number of timestamps")
        
        if not self.frames_analysis:
            raise ValueError("No frame analysis data provided")

@dataclass
class Video:
    """Data class to video information"""
    
    video_path: str
    video_id: Optional[str] = None
    frames: Optional[List[Frame]] = None
    analysis: Optional[FramesAnalysisResult] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        assert isinstance(self.video_path, str), "video_path must be a string"

        vr = VideoReader(self.video_path)
        self.video_duration = vr._num_frame / vr.get_avg_fps()
        self.video_fps = vr.get_avg_fps()
        self.video_size = vr.next().shape

        if self.video_id is None:
            self.video_id = str(uuid.uuid4())

class ActivityType(Enum):
    """Enumeration of recognizable activities"""
    UNKNOWN = "unknown"
    NORMAL_WALKING = "normal_walking"
    RUNNING = "running"
    FIGHTING = "fighting"
    GROUP_GATHERING = "group_gathering"
    MILITIA_BEHAVIOR = "militia_behavior"
    VEHICLE_DRIVING = "vehicle_driving"
    VEHICLE_SPEEDING = "vehicle_speeding"
    VEHICLE_STOPPING = "vehicle_stopping"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    CROWD_FORMATION = "crowd_formation"

ACTIVITY_PROMPTS = {
    ActivityType.NORMAL_WALKING: "A person walking normally",
    ActivityType.RUNNING: "A person running or jogging",
    ActivityType.FIGHTING: "People fighting or engaging in physical conflict",
    ActivityType.GROUP_GATHERING: "A group of people gathering together",
    ActivityType.MILITIA_BEHAVIOR: "People in military or militia formation",
    ActivityType.VEHICLE_DRIVING: "A vehicle or multiple vehicles driving on the road",
    ActivityType.VEHICLE_SPEEDING: "A vehicle or multiple vehicles driving at high speed",
    ActivityType.VEHICLE_STOPPING: "A vehicle or multiple vehicles stopping or parked",
    ActivityType.SUSPICIOUS_BEHAVIOR: "Suspicious or unusual behavior",
    ActivityType.CROWD_FORMATION: "A crowd forming or moving",
    ActivityType.UNKNOWN: "Unknown or unclear activity"
}


@dataclass
class ActivityClassificationResult:
    """Data class for activity recognition results"""
    activity_type: ActivityType
    confidence: float