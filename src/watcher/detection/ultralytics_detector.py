import os
import time
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import cv2
import torch
from ultralytics import YOLOE
from ultralytics.engine.results import Results, Boxes
import supervision as sv

import warnings

from .base import DroneObjectDetector,logger
from ..base import Frame, Detection, ACTIVITY_PROMPTS



# Suppress ultralytics warnings
warnings.filterwarnings("ignore", category=UserWarning)


class YOLOModel:
    """
    YOLO model wrapper that implements the Model interface from base.py
    """
    
    def __init__(self, 
        model_path: str, 
        device: str = "cpu", 
        input_size: Tuple[int, int] = (640, 640), 
        confidence_threshold: float = 0.3,
        batch_size: int = 8,
        verbose: bool = False,
        max_det: int = 50,):
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
        self.label_map = {}
        self.verbose = verbose
        # load model and labelmap
        self._load_model()
        
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLOE(self.model_path)
            names = list(ACTIVITY_PROMPTS.values())
            self.model.set_classes(names, self.model.get_text_pe(names))
            self.label_map = self.model.names
            self.model.to(self.device)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _predict(self,source)->Results:
        return self.model.predict(source, 
        verbose=self.verbose, 
        imgsz=max(self.input_size), 
        conf=self.confidence_threshold, 
        max_det=self.max_det, 
        device=self.device)
    
    def _sliced_inference(self, image: np.ndarray,iou_threshold=0.5,thread_workers:int=1,return_sv_detections:bool=False) -> dict:
        """
        Run inference on a list of frames in a batched manner.
        """
        #box_annotator = sv.BoxAnnotator()
        #label_annotator = sv.LabelAnnotator()

        def callback(image: np.ndarray) -> sv.Detections:
            result = self._predict(image)[0]
            return sv.Detections.from_ultralytics(result)
        
        slicer = sv.InferenceSlicer(callback = callback,slice_wh=self.input_size,
                                    iou_threshold=iou_threshold,
                                    thread_workers=thread_workers,
                                    overlap_ratio_wh=None,
                                    overlap_wh=(100,100)
                                    )
        detections = slicer(image)
        #annotated_image = box_annotator.annotate(image.copy(), detections=detections)
        #annotated_image = label_annotator.annotate(annotated_image, detections=detections)
        output = dict(bbox=detections.xyxy,
                      label=detections.class_id,
                      confidence=detections.confidence)
        if return_sv_detections:
            return detections
        return output
    
    def _inference(self, image: np.ndarray) -> dict:
        # Run inference
        results = self._predict(image)

        detections = sv.Detections.from_ultralytics(results)
        output = dict(bbox=detections.xyxy,
                      label=detections.class_id,
                      confidence=detections.confidence)
        return output
    
    def __call__(self, frames: List[Frame],sliced:bool=False) -> List[Frame]:
        """
        Process frames through YOLO model
        
        Args:
            frames: List of Frame objects
            
        Returns:
            List of Frame objects with detections added
        """
        if not frames:
            raise RuntimeError("YOLO model not loaded")
        
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        
        # Extract frame images
        
        if sliced:
            results = [self._sliced_inference(frame.image) for frame in frames]
        else:
            results = [self._inference(frame.image) for frame in frames]
        
        # Process results
        for frame, result in zip(frames, results):
            frame.detections = self._process_detections(result, frame.timestamp)
        
        return frames
    
    def _process_detections(self, result: dict, timestamp: float) -> List[Detection]:
        """
        Convert YOLO results to Detection objects
        
        Args:
            result: YOLO Results object
            timestamp: Frame timestamp
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Get boxes data
        boxes = result["bbox"]
        labels = result["label"]
        confidences = result["confidence"]
                 
        # Extract detection data
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.tolist()
            
            # Get class info
            class_id = int(labels[i])
            confidence = float(confidences[i])
                        
            # Create Detection object
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                class_name=self.label_map[class_id],
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
        self.max_det = max_det
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = self._load_model(model_path)
        self.label_map = self.model.label_map
            
        logger.info(f"Using device: {self.device}")

        # Performance tracking
        self.reset_stats()

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
    
    def inference(self,frames: List[Frame],sliced:bool=False):
        assert isinstance(frames,list), "frames must be a list"
        try:
            detections = self.model(frames,sliced=sliced)
            return detections
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during inference: {e}")
            raise
    
    def annotate_frames(self,frames: List[Frame],sliced:bool=True) -> List[Frame]:
        assert isinstance(frames,list), "frames must be a list"
        try:
            box_annotator = sv.BoxAnnotator()
            for frame in frames:
                if sliced:
                    detections = self.model._sliced_inference(frame.image,return_sv_detections=True)
                else:
                    detections = self.model._predict(frame.image)[0]
                    detections = sv.Detections.from_ultralytics(detections)
                annotated_image = box_annotator.annotate(
                scene=frame.image, detections=detections)
                frame.image = annotated_image
            return frames
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during inference: {e}")
            raise

    def inference_video(self, video_path: str,output_path: str,sliced:bool=False) -> None:
        """
        Run inference on a video
        """
        box_annotator = sv.BoxAnnotator()
        tracker = sv.ByteTrack()
        #trace_annotator = sv.TraceAnnotator()

        def callback(image: np.ndarray, _: int) -> np.ndarray:
            if sliced:
                detections = self.model._sliced_inference(image,return_sv_detections=True)
            else:
                detections = self.model._predict(image)[0]
                detections = sv.Detections.from_ultralytics(detections)
            detections = tracker.update_with_detections(detections)
            annotated_frame = box_annotator.annotate(
                image.copy(), detections=detections)
            #annotated_frame =  trace_annotator.annotate(
            #    annotated_frame, detections=detections)
            return annotated_frame

        sv.process_video(
            source_path=video_path,
            target_path=output_path,
            callback=callback
        )

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
