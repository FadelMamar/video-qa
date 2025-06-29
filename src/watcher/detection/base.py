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
logger = logging.getLogger(__name__)

from ..base import Frame

class Model(ABC):
    @abstractmethod
    def __call__(self, frames: List[Frame]) -> List[Frame]:
        pass

class DroneObjectDetector(ABC):
    """
    Object detection model for drone footage analysis
    
    Supports YOLO and similar models for detecting humans, cars, and motorcycles
    with efficient batching and GPU acceleration when available.
    """
        
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        batch_size: int = 8,
        input_size: Tuple[int, int] = (640, 640),
        label_map: Dict[int, str] = {0: 'person', 1: 'car', 2: 'motorcycle', 3: 'bicycle', 4: 'bus', 5: 'truck'}
    ):
        """
        Initialize the drone object detector
        
        Args:
            model_path: Path to the  model file
            confidence_threshold: Minimum confidence for detections
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for inference
            input_size: Input image size for the model
            label_map: Dictionary mapping class IDs to class names
        """
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.input_size = input_size
        self.label_map = label_map
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    def _load_model(self, model_path: str)->Model:
        pass
    
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
    
    def detect_objects_batch(
        self,
        frames: List[Frame],
    ) -> List[Frame]:
        """
        Perform object detection on a batch of frames
        
        Args:
            frames: List of BGR frames from OpenCV
            frame_timestamps: Timestamp for each frame
            frame_ids: Frame ID for each frame
            
        Returns:
            List of FrameResult objects with detections
        """

        if not frames:
            return frames
        
        for frame in frames:
            assert isinstance(frame, Frame), "Frames must be a list of Frame objects"
        
        start_time = time.time()
        
        try:
            # Process frames through model
            processed_frames = self.model(frames)
            
            # Postprocess results
            processed_frames = self.postprocess_results(processed_frames)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_frames_processed += len(frames)
            self.total_processing_time += processing_time
            
            logger.debug(f"Processed {len(frames)} frames in {processing_time:.3f}s")
            return processed_frames
            
        except Exception as e:
            logger.error(f"Error in batch detection: {e}")
            raise
        
            
    def detect_objects_single(
        self,
        frame: Frame,
    ) -> Frame:
        """
        Perform object detection on a single frame
        
        Args:
            frame: BGR frame from OpenCV
            frame_timestamp: Timestamp of the frame
            frame_id: ID of the frame
            
        Returns:
            FrameResult object with detections
        """
        return self.detect_objects_batch([frame])[0]
