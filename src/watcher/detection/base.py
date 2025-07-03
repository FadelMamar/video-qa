import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import cv2
import torch
from abc import ABC, abstractmethod
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Detection")

from ..base import Frame


class DroneObjectDetector(ABC):
    """
    Object detection model for drone footage analysis
    
    Supports YOLO and similar models for detecting humans, cars, and motorcycles
    with efficient batching and GPU acceleration when available.
    """
    
    @abstractmethod
    def preprocess_frames(self, frames: List[Frame]):
        """
        Preprocess frames for model inference
        
        Args:
            frames: List of BGR frames from OpenCV
            
        Returns:
            Preprocessed tensor ready for inference
        """
        pass
    
    def postprocess_results(
        self,
        frame_list:List[Frame],
    ) -> List[Frame]:
        """
        Postprocess model results into structured format
        
        Args:
            results: Raw model output
            original_shapes: Original frame dimensions (height, width)
            frame_timestamps: Timestamp for each frame
            frame_ids: Frame ID for each frame
            
        Returns:
            List of FrameResult objects
        """
        frames = deepcopy(frame_list)
        for i, frame in enumerate(frames):
            detections = []
                                                                            
            for detection in frame.detections:
                if detection.confidence >= self.confidence_threshold and detection.class_name.lower() in self.label_map.values():
                    detections.append(detection)
            
            frame.detections = detections
        
        return frames
    
    @abstractmethod
    def inference_video(self, video_path: str,output_path: str) -> None:
        pass
    
    @abstractmethod
    def inference(self,frames: List[Frame],sliced:bool=False):
        pass
    
    
    