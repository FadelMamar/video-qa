import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Mapping
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import deque, defaultdict
from pathlib import Path
import json
from enum import Enum

from ..base import Frame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        ActivityType.VEHICLE_DRIVING: "A vehicle driving on the road",
        ActivityType.VEHICLE_SPEEDING: "A vehicle driving at high speed",
        ActivityType.VEHICLE_STOPPING: "A vehicle stopping or parked",
        ActivityType.SUSPICIOUS_BEHAVIOR: "Suspicious or unusual behavior",
        ActivityType.CROWD_FORMATION: "A large crowd forming or moving",
        ActivityType.UNKNOWN: "Unknown or unclear activity"
            }

@dataclass
class TrackedObject:
    """Data class for tracked object information"""
    track_id: int
    class_name: str
    bboxes: List[Tuple[float, float, float, float]]  # Sequence of bounding boxes
    confidences: List[float]
    timestamps: List[float]
    frame_ids: List[int]
    
    def __post_init__(self):
        """Ensure all lists have the same length"""
        lengths = [len(self.bboxes), len(self.confidences), len(self.timestamps), len(self.frame_ids)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All tracking data lists must have the same length")

@dataclass
class ActivityResult:
    """Data class for activity recognition results"""
    track_id: int
    activity_type: ActivityType
    confidence: float
    start_timestamp: float
    end_timestamp: float
    bounding_region: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    object_class: str = ""
    frame_sequence_length: int = 0

@dataclass
class GroupActivityResult:
    """Data class for group activity recognition results"""
    group_id: str
    track_ids: List[int]
    activity_type: ActivityType
    confidence: float
    start_timestamp: float
    end_timestamp: float
    center_point: Tuple[float, float]
    group_size: int
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)


class ActivityClassifier(ABC):
    """Activity classification module"""
    
    def __init__(self, model_path: str, 
                device: str = "cpu",
                confidence_threshold: float = 0.5,
                ambiguous_threshold: float = 0.3,):

        self.device = device   
        self.confidence_threshold = confidence_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.classifier =  self.load_model(model_path)

        assert self.ambiguous_threshold < self.confidence_threshold, "Ambiguous threshold must be less than confidence threshold"

        # Activity type mapping
        self.activity_types = list(ActivityType)
        self.activity_to_idx = {act: idx for idx, act in enumerate(self.activity_types)}
        self.idx_to_activity = {idx: act for act, idx in self.activity_to_idx.items()}

    def __call__(self, images: List[np.ndarray],obj:TrackedObject) -> List[ActivityResult]:
        preprocessed_images = self.preprocess(images)
        activities = self.classifier(preprocessed_images)
        return self.postprocess(activities,)
    
        
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass
    
    @abstractmethod
    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        pass


    def postprocess(self, obj:TrackedObject) -> ActivityResult:

        if not obj.bboxes:
            bounding_region = (0, 0, 0, 0)
        else:
            x1_min = min(bbox[0] for bbox in obj.bboxes)
            y1_min = min(bbox[1] for bbox in obj.bboxes)
            x2_max = max(bbox[2] for bbox in obj.bboxes)
            y2_max = max(bbox[3] for bbox in obj.bboxes)
            bounding_region = (x1_min, y1_min, x2_max, y2_max)
        
        # Apply confidence thresholds
        supporting_evidence: Dict[str, Any] = {}
        activity_type = self.idx_to_activity[obj.class_name]
        confidence = np.mean(obj.confidences)
        if confidence < self.ambiguous_threshold:
            activity_type = ActivityType.UNKNOWN
            supporting_evidence = {"ambiguous": True, "raw_confidence": confidence}
        elif confidence < self.confidence_threshold:
            # Mark as ambiguous but keep prediction
            supporting_evidence = {"ambiguous": True, "raw_confidence": confidence}
        else:
            supporting_evidence = {"confident": True, "raw_confidence": confidence}
        
        # Create result
        result = ActivityResult(
            track_id=obj.track_id,
            activity_type=activity_type,
            confidence=confidence,
            start_timestamp=obj.timestamps[0],
            end_timestamp=obj.timestamps[-1],
            bounding_region=bounding_region,
            supporting_evidence=supporting_evidence,
            object_class=obj.class_name,
            frame_sequence_length=len(obj.bboxes)
        )

        return result
    


    """
    Create TrackedObject instances from detection and tracking results
    
    Args:
        detection_results: Results from object detection
        tracking_results: Results from object tracking
        min_sequence_length: Minimum sequence length for activity recognition
        
    Returns:
        List of TrackedObject instances
    """
    tracked_objects: List[TrackedObject] = []
    
    # Group detections by track ID
    track_data: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        'bboxes': [], 'confidences': [], 'timestamps': [], 
        'frame_ids': [], 'class_name': None
    })
    
    # Process tracking results (assuming they contain track_id, detection info)
    for track_result in tracking_results:
        track_id = track_result.get('track_id')
        if track_id is None:
            continue
        
        track_data[track_id]['bboxes'].append(track_result['bbox'])
        track_data[track_id]['confidences'].append(track_result['confidence'])
        track_data[track_id]['timestamps'].append(track_result['timestamp'])
        track_data[track_id]['frame_ids'].append(track_result['frame_id'])
        
        if track_data[track_id]['class_name'] is None:
            track_data[track_id]['class_name'] = track_result['class_name']
    
    # Create TrackedObject instances
    for track_id, data in track_data.items():
        if len(data['bboxes']) >= min_sequence_length:
            tracked_obj = TrackedObject(
                track_id=track_id,
                class_name=data['class_name'] or "unknown",
                bboxes=data['bboxes'],
                confidences=data['confidences'],
                timestamps=data['timestamps'],
                frame_ids=data['frame_ids']
            )
            tracked_objects.append(tracked_obj)
    
    return tracked_objects