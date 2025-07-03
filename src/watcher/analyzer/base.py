import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Union, Dict, Any
import numpy as np
import logging
from PIL import Image
import io
from enum import Enum

from ..base import Video

logger = logging.getLogger("Analyzer")

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

@dataclass
class FramesAnalysisResult:
    """Container for video analysis results."""
    video: Video
    frames_analysis: List[str]
    timestamps: List[float]
    summary: str
    
    def __post_init__(self):
        """Validate the analysis result data."""
        if len(self.frames_analysis) != len(self.timestamps):
            raise ValueError("Number of frame analyses must match number of timestamps")
        
        if not self.frames_analysis:
            raise ValueError("No frame analysis data provided")


class ActivityClassifier(ABC):
    """Activity classification module"""
    
    def __init__(self, model_path: str, 
                device: str = "cpu",
                confidence_threshold: float = 0.5,):

        self.device = device   
        self.confidence_threshold = confidence_threshold
        self.classifier =  self.load_model(model_path)

        # Activity type mapping
        self.activity_types = list(ActivityType)
        self.activity_to_idx = {act: idx for idx, act in enumerate(self.activity_types)}
        self.idx_to_activity = {idx: act for act, idx in self.activity_to_idx.items()}
        self.activity_prompts = ACTIVITY_PROMPTS

    def __call__(self, images: List[np.ndarray]) -> List[ActivityClassificationResult]:
        preprocessed_images = self.preprocess(images)
        activities = self.classifier(preprocessed_images)
        return self.postprocess(activities,)
    
        
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass
    
    @abstractmethod
    def preprocess(self, images: List[np.ndarray]):
        pass

    @abstractmethod
    def postprocess(self, activities: List[ActivityClassificationResult]) -> List[ActivityClassificationResult]:
        """
        Postprocess the activities to apply confidence thresholds
        """
        pass


class BaseAnalyzer(ABC):
    """Abstract base class for video frame analyzers."""
    
    @abstractmethod
    def analyze(self, images: Union[bytes, List[bytes]]) -> str:
        """
        Analyze a sequence of images.
        
        Args:
            images: Single image bytes or list of image bytes
            
        Returns:
            Analysis result as string
        """
        pass
    
    def analyze_activity(
        self, 
        images: Union[bytes, List[bytes]], 
        activity_type: Optional[ActivityType] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze images for specific activity recognition.
        
        Args:
            images: Single image bytes or list of image bytes
            activity_type: Specific activity type to look for
            context: Additional context information (e.g., tracked object info)
            
        Returns:
            Activity analysis result as string
        """
        # Default implementation falls back to general analysis
        return self.analyze(images)
        
    def _calculate_bounding_region(self, bboxes: List[tuple]) -> tuple:
        """Calculate overall bounding region from sequence of bounding boxes."""
        if not bboxes:
            return (0, 0, 0, 0)
        
        x1_min = min(bbox[0] for bbox in bboxes)
        y1_min = min(bbox[1] for bbox in bboxes)
        x2_max = max(bbox[2] for bbox in bboxes)
        y2_max = max(bbox[3] for bbox in bboxes)
        
        return (x1_min, y1_min, x2_max, y2_max)
    
    @staticmethod
    def _encode_image(image: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image).decode("utf-8")
    
    @staticmethod
    def _validate_images(images: Union[bytes, List[bytes]]) -> List[bytes]:
        """
        Validate and normalize image input.
        
        Args:
            images: Single image or list of images
            
        Returns:
            List of image bytes
            
        Raises:
            ValueError: If images are not in expected format
        """
        if isinstance(images, bytes):
            return [images]
        
        if isinstance(images, list):
            for img in images:
                if not isinstance(img, bytes):
                    raise ValueError(f"Expected bytes, got {type(img)}")
            return images
        
        raise ValueError(f"Expected bytes or list[bytes], got {type(images)}")

    def load_image_as_pil(self, image: bytes) -> Image.Image:
        """Load image as PIL Image."""
        return Image.open(io.BytesIO(image))


class ModelConfig:
    """Configuration for VLM model settings."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        model_type: Literal["chat", "text"] = "chat",
        cache: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.model_type = model_type
        self.cache = cache
        self.api_base = api_base or os.environ.get("BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-no-key-required")
    
    def validate(self) -> None:
        """Validate the model configuration."""
        if not self.model_name:
            raise ValueError("Model name must be specified")
        
        if not self.model_name.startswith("openai/"):
            raise ValueError("Only OpenAI compatible endpoints are supported")
        
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.model_type not in ["chat", "text"]:
            raise ValueError("Model type must be 'chat' or 'text'")