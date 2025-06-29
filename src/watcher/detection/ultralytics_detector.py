import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import warnings

from .base import Model, DroneObjectDetector
from ..base import Frame, Detection

# Configure logging
logger = logging.getLogger(__name__)

# Suppress ultralytics warnings
warnings.filterwarnings("ignore", category=UserWarning)


class YOLOModel(Model):
    """
    YOLO model wrapper that implements the Model interface from base.py
    """
    
    def __init__(self, 
        model_path: str, 
        device: str = "cpu", 
        input_size: Tuple[int, int] = (640, 640), 
        confidence_threshold: float = 0.3,
        batch_size: int = 8,
        max_det: int = 300,):
        """
        Initialize YOLO model
        
        Args:
            model_path: Path to YOLO model file (.pt, .onnx, etc.)
            device: Device to use ('cpu', 'cuda', or 'auto')
            input_size: Input image size for the model
            confidence_threshold: Minimum confidence for detections
            batch_size: Batch size for inference
            max_det: Maximum number of detections per image
        """
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.max_det = max_det
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def __call__(self, frames: List[Frame]) -> List[Frame]:
        """
        Process frames through YOLO model
        
        Args:
            frames: List of Frame objects
            
        Returns:
            List of Frame objects with detections added
        """
        if not frames:
            return frames
        
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        # Extract frame images
        frame_images = [frame.image for frame in frames]
        
        # Run inference
        results = self.model(frame_images, 
        verbose=False, 
        batch=self.batch_size,
        imgsz=self.input_size, 
        conf=self.confidence_threshold, 
        max_det=self.max_det, 
        device=self.device)
        
        # Process results
        for frame, result in zip(frames, results):
            frame.detections = self._process_detections(result, frame.timestamp)
        
        return frames
    
    def _process_detections(self, result: Results, timestamp: float) -> List[Detection]:
        """
        Convert YOLO results to Detection objects
        
        Args:
            result: YOLO Results object
            timestamp: Frame timestamp
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if result.boxes is None:
            return detections
        
        # Get boxes data
        boxes = result.boxes
         
        if len(boxes) == 0:
            return detections
        
        # Extract detection data
        for i in range(len(boxes)):
            # Get box coordinates (x1, y1, x2, y2)
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box.tolist()
            
            # Get class info
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            
            # Get class name
            class_name = result.names.get(class_id, f"class_{class_id}")
            
            # Create Detection object
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                frame_timestamp=timestamp
            )
            
            detections.append(detection)
        
        return detections


class YOLODetector(DroneObjectDetector):
    """
    YOLO-based object detector for drone footage analysis
    
    Uses ultralytics YOLO models for detecting objects in aerial video frames.
    Supports various YOLO model formats (PyTorch, ONNX, etc.) and provides
    efficient batching and GPU acceleration.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        batch_size: int = 8,
        input_size: Tuple[int, int] = (640, 640),
        label_map: Optional[Dict[int, str]] = None,
        model_type: str = "yolo8",  # yolo8, yolo5, etc.
        max_det: int = 300,  # Maximum number of detections per image
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file (.pt, .onnx, etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for inference
            input_size: Input image size for the model
            label_map: Dictionary mapping class IDs to class names (optional)
            model_type: Type of YOLO model (yolo8, yolo5, etc.)
            max_det: Maximum number of detections per image
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.input_size = input_size
        self.model_type = model_type
        self.max_det = max_det
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Set label map
        if label_map is None:
            # Default COCO classes for drone surveillance
            self.label_map = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
                49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                79: 'toothbrush'
            }
        else:
            self.label_map = label_map
        
        # Performance tracking
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
        
        # Initialize model
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> YOLOModel:
        """
        Load YOLO model
        
        Args:
            model_path: Path to model file
            
        Returns:
            YOLOModel instance
        """
        try:
            # Validate model path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create YOLO model wrapper
            model = YOLOModel(model_path, device=self.device, input_size=self.input_size, 
            confidence_threshold=self.confidence_threshold, 
            batch_size=self.batch_size, 
            max_det=self.max_det)
            logger.info(f"Successfully loaded YOLO model: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def preprocess_frames(self, frames: List[Frame]) -> List[Frame]:
        """
        Preprocess frames for YOLO inference
        
        Args:
            frames: List of BGR frames from OpenCV
            
        Returns:
            List of preprocessed frames
        """
        # YOLO handles preprocessing internally, so we just return the frames
        # The model will handle resizing, normalization, etc.
        return frames
    
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = (self.total_processing_time / self.total_frames_processed 
                   if self.total_frames_processed > 0 else 0)
        
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_processing_time": self.total_processing_time,
            "average_time_per_frame": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "device": self.device,
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_frames_processed = 0
        self.total_processing_time = 0.0


# Convenience function for creating detector
def create_yolo_detector(
    model_path: str,
    confidence_threshold: float = 0.3,
    device: Optional[str] = None,
    **kwargs
) -> YOLODetector:
    """
    Convenience function to create a YOLO detector
    
    Args:
        model_path: Path to YOLO model file
        confidence_threshold: Minimum confidence for detections
        device: Device to use
        **kwargs: Additional arguments for YOLODetector
        
    Returns:
        YOLODetector instance
    """
    return YOLODetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
        **kwargs
    )
