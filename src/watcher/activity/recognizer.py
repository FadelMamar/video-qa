"""
Activity Recognition Module for Drone Footage Analysis

This module provides activity recognition capabilities for tracked objects in drone footage,
focusing on identifying specific human and vehicle activities using vision-language models
and action recognition techniques.
"""
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
from .base import ActivityClassifier, ActivityResult, GroupActivityResult, TrackedObject, ActivityType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DroneActivityRecognizer:
    """
    Activity recognition system for drone footage analysis
    
    Recognizes specific activities from tracked object sequences with focus on
    security-relevant behaviors and vehicle activities.
    """
    
    def __init__(
        self,
        activity_classifier: ActivityClassifier,
        sequence_length: int = 16,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the activity recognizer
        
        Args:
            model_path: Path to pretrained model weights
            confidence_threshold: Minimum confidence for activity classification
            ambiguous_threshold: Threshold below which activities are marked as ambiguous
            sequence_length: Number of frames to analyze per clip
            device: Device to use ('cpu' or 'cuda', if None then auto)
            input_size: Input image size for the model
        """
        self.sequence_length = sequence_length
        self.input_size = input_size      

        self.activity_classifier = activity_classifier
        
        # Tracking data storage
        self.object_tracks: Dict[int, deque] = defaultdict(deque)
        self.group_tracks: Dict[str, List] = defaultdict(list)
            
    
    def extract_object_clips(
        self,
        frames: List[Frame],
        tracked_objects: List[TrackedObject]
    ) -> Dict[int, List[List[np.ndarray]]]:
        """
        Extract object-centered clips from frames
        
        Args:
            frames: List of video frames
            tracked_objects: List of tracked objects
            
        Returns:
            Dictionary mapping track_id to list of cropped object clips
        """
        object_clips: Dict[int, List[List[np.ndarray]]] = defaultdict(list)
        
        for obj in tracked_objects:
            if len(obj.bboxes) < self.sequence_length:
                continue
                
            # Extract clips for this object
            for i in range(len(obj.bboxes) - self.sequence_length + 1):
                clip_frames: List[np.ndarray] = []
                
                for j in range(i, i + self.sequence_length):
                    frame_idx = obj.frame_ids[j]
                    if frame_idx < len(frames):
                        frame = frames[frame_idx]
                        bbox = obj.bboxes[j]
                        
                        # Crop object region with padding
                        cropped = self._crop_object_region(frame, bbox)
                        clip_frames.append(cropped)
                
                if len(clip_frames) == self.sequence_length:
                    object_clips[obj.track_id].append(clip_frames)
        
        return object_clips
    
    def _crop_object_region(
        self,
        frame: Frame,
        bbox: Tuple[float, float, float, float],
        padding_factor: float = 0.2
    ) -> np.ndarray:
        """
        Crop object region from frame with padding
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            padding_factor: Padding around the bounding box
            
        Returns:
            Cropped and resized frame
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.image.shape[:2]
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        
        pad_w = width * padding_factor
        pad_h = height * padding_factor
        
        # Expand bbox with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop region
        cropped = frame.image[int(y1):int(y2), int(x1):int(x2)]
        
        # Resize to input size
        if cropped.size > 0:
            resized = cv2.resize(cropped, self.input_size,interpolation=cv2.INTER_LINEAR)  # type: ignore
        else:
            # Handle empty crop
            resized = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        
        return resized
    
    def preprocess_clips(self, clip: List[Frame]) -> List[np.ndarray]:
        """
        Preprocess clips for model input
        
        Args:
            clips: List of clips, each clip is a list of frames
            
        Returns:
            Preprocessed tensor of shape (batch_size, sequence_length, channels, height, width)
        """
        
        clip_np = []
        
        for frame in clip:
            rgb_frame = frame.image.copy().astype(np.float32)
            clip_np.append(rgb_frame)
                
        # Stack into batch tensor
        return clip_np
    
    @torch.no_grad()
    def classify_activities(
        self,
        object_clips: Dict[int, List[List[Frame]]],
        tracked_objects: Dict[int, TrackedObject]
    ) -> List[ActivityResult]:
        """
        Classify activities for tracked objects
        
        Args:
            object_clips: Dictionary mapping track_id to clips
            tracked_objects: Dictionary mapping track_id to TrackedObject
            
        Returns:
            List of ActivityResult objects
        """
        results: List[ActivityResult] = []
        
        for track_id, clip in object_clips.items():
            if not clip or track_id not in tracked_objects:
                continue
            
            obj = tracked_objects[track_id]
            
            # Process clips in batches
            start_time = time.time()
            
            # Preprocess clip
            clip_np = self.preprocess_clips(clip)
            
            # Extract features
            result = self.activity_classifier(clip_np,obj)
            
            results.append(result)
            
            # Update performance metrics
            self.total_clips_processed += 1
            self.total_processing_time += time.time() - start_time
        
        return results
    
    
    def detect_group_activities(
        self,
        individual_results: List[ActivityResult],
        tracked_objects: Dict[int, TrackedObject],
        proximity_threshold: float = 100.0
    ) -> List[GroupActivityResult]:
        """
        Detect group activities based on individual results and proximity
        
        Args:
            individual_results: List of individual activity results
            tracked_objects: Dictionary of tracked objects
            proximity_threshold: Distance threshold for group formation
            
        Returns:
            List of GroupActivityResult objects
        """
        group_results: List[GroupActivityResult] = []
        
        # Group objects by timestamp and proximity
        timestamp_groups: Dict[int, List[ActivityResult]] = defaultdict(list)
        
        for result in individual_results:
            # Round timestamp to nearest second for grouping
            timestamp_key = round(result.start_timestamp)
            timestamp_groups[timestamp_key].append(result)
        
        # Analyze each timestamp group
        for timestamp, results in timestamp_groups.items():
            if len(results) < 2:
                continue
            
            # Find spatial clusters
            clusters = self._find_spatial_clusters(results, tracked_objects, proximity_threshold)
            
            for cluster in clusters:
                if len(cluster) >= 2:  # Minimum group size
                    group_activity = self._analyze_group_behavior(cluster, tracked_objects)
                    if group_activity:
                        group_results.append(group_activity)
        
        return group_results
    
    def _find_spatial_clusters(
        self,
        results: List[ActivityResult],
        tracked_objects: Dict[int, TrackedObject],
        proximity_threshold: float
    ) -> List[List[ActivityResult]]:
        """Find spatial clusters of objects"""
        clusters: List[List[ActivityResult]] = []
        used_results: set[int] = set()
        
        for i, result in enumerate(results):
            if i in used_results:
                continue
            
            cluster = [result]
            used_results.add(i)
            
            # Get center point of current object
            center1 = self._get_object_center(tracked_objects[result.track_id])
            
            # Find nearby objects
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_results:
                    continue
                
                center2 = self._get_object_center(tracked_objects[other_result.track_id])
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance <= proximity_threshold:
                    cluster.append(other_result)
                    used_results.add(j)
            
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def _get_object_center(self, obj: TrackedObject) -> Tuple[float, float]:
        """Get center point of tracked object"""
        if not obj.bboxes:
            return (0, 0)
        
        # Use the last bounding box
        bbox = obj.bboxes[-1]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        return (center_x, center_y)
    
    def _analyze_group_behavior(
        self,
        cluster: List[ActivityResult],
        tracked_objects: Dict[int, TrackedObject]
    ) -> Optional[GroupActivityResult]:
        """Analyze behavior of a group of objects"""
        
        # Get group characteristics
        group_size = len(cluster)
        track_ids = [result.track_id for result in cluster]
        
        # Calculate group center
        centers = [self._get_object_center(tracked_objects[tid]) for tid in track_ids]
        group_center = (
            sum(c[0] for c in centers) / len(centers),
            sum(c[1] for c in centers) / len(centers)
        )
        
        # Analyze individual activities
        individual_activities = [result.activity_type for result in cluster]
        activity_counts: Dict[ActivityType, int] = defaultdict(int)
        
        for activity in individual_activities:
            activity_counts[activity] += 1
        
        # Determine group activity
        group_activity = ActivityType.UNKNOWN
        confidence = 0.0
        evidence: Dict[str, Any] = {}
        
        # Check for specific group patterns
        if group_size >= 3:
            # Check for militia-like behavior
            if (activity_counts[ActivityType.SUSPICIOUS_BEHAVIOR] >= 2 or
                activity_counts[ActivityType.GROUP_GATHERING] >= 2):
                group_activity = ActivityType.MILITIA_BEHAVIOR
                confidence = min(0.8, sum(r.confidence for r in cluster) / len(cluster))
                evidence["pattern"] = "organized_group_behavior"
            
            # Check for crowd formation
            elif group_size >= 5:
                group_activity = ActivityType.CROWD_FORMATION
                confidence = 0.7
                evidence["pattern"] = "large_group_formation"
        
        # Check for fighting behavior
        if activity_counts[ActivityType.FIGHTING] >= 2:
            group_activity = ActivityType.FIGHTING
            confidence = max(r.confidence for r in cluster if r.activity_type == ActivityType.FIGHTING)
            evidence["pattern"] = "group_conflict"
        
        if group_activity != ActivityType.UNKNOWN:
            return GroupActivityResult(
                group_id=f"group_{hash(tuple(sorted(track_ids)))}",
                track_ids=track_ids,
                activity_type=group_activity,
                confidence=confidence,
                start_timestamp=min(r.start_timestamp for r in cluster),
                end_timestamp=max(r.end_timestamp for r in cluster),
                center_point=group_center,
                group_size=group_size,
                supporting_evidence=evidence
            )
        
        return None
    
    def process_tracked_objects(
        self,
        frames: List[Frame],
        tracked_objects: List[TrackedObject]
    ) -> Tuple[List[ActivityResult], List[GroupActivityResult]]:
        """
        Process tracked objects for activity recognition
        
        Args:
            frames: List of video frames
            tracked_objects: List of tracked objects
            
        Returns:
            Tuple of (individual_results, group_results)
        """
        logger.info(f"Processing {len(tracked_objects)} tracked objects")
        
        # Convert to dictionary for easier access
        tracked_objects_dict = {obj.track_id: obj for obj in tracked_objects}
        
        # Extract object clips
        object_clips = self.extract_object_clips(frames, tracked_objects)
        
        # Classify individual activities
        individual_results = self.classify_activities(object_clips, tracked_objects_dict)
        
        # Detect group activities
        group_results = self.detect_group_activities(
            individual_results, tracked_objects_dict
        )
        
        logger.info(f"Found {len(individual_results)} individual activities")
        logger.info(f"Found {len(group_results)} group activities")
        
        return individual_results, group_results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.total_clips_processed == 0:
            return {"avg_processing_time": 0.0, "clips_per_second": 0.0}
        
        avg_time = self.total_processing_time / self.total_clips_processed
        clips_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "total_clips": self.total_clips_processed,
            "total_time": self.total_processing_time,
            "avg_processing_time": avg_time,
            "clips_per_second": clips_per_sec
        }
    
    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        ambiguous_threshold: Optional[float] = None
    ) -> None:
        """Update confidence thresholds"""
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        
        if ambiguous_threshold is not None:
            self.ambiguous_threshold = max(0.0, min(1.0, ambiguous_threshold))
        
        logger.info(f"Updated thresholds - confidence: {self.confidence_threshold}, "
                   f"ambiguous: {self.ambiguous_threshold}")

# Utility functions

def save_activity_results(
    individual_results: List[ActivityResult],
    group_results: List[GroupActivityResult],
    output_path: str
) -> None:
    """Save activity recognition results to JSON file"""
    
    def serialize_activity_result(result: ActivityResult) -> Dict[str, Any]:
        return {
            "track_id": result.track_id,
            "activity_type": result.activity_type.value,
            "confidence": result.confidence,
            "start_timestamp": result.start_timestamp,
            "end_timestamp": result.end_timestamp,
            "bounding_region": result.bounding_region,
            "supporting_evidence": result.supporting_evidence,
            "object_class": result.object_class,
            "frame_sequence_length": result.frame_sequence_length
        }
    
    def serialize_group_result(result: GroupActivityResult) -> Dict[str, Any]:
        return {
            "group_id": result.group_id,
            "track_ids": result.track_ids,
            "activity_type": result.activity_type.value,
            "confidence": result.confidence,
            "start_timestamp": result.start_timestamp,
            "end_timestamp": result.end_timestamp,
            "center_point": result.center_point,
            "group_size": result.group_size,
            "supporting_evidence": result.supporting_evidence
        }
    
    data: Dict[str, Any] = {
        "individual_activities": [serialize_activity_result(r) for r in individual_results],
        "group_activities": [serialize_group_result(r) for r in group_results],
        "summary": {
            "total_individual_activities": len(individual_results),
            "total_group_activities": len(group_results),
            "activity_type_counts": {}
        }
    }
    
    # Count activity types
    for result in individual_results:
        activity_name = result.activity_type.value
        data["summary"]["activity_type_counts"][activity_name] = \
            data["summary"]["activity_type_counts"].get(activity_name, 0) + 1
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Activity results saved to {output_path}")
