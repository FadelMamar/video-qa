import base64
import os
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union, Dict, Any
import numpy as np
import logging
from PIL import Image
import io

from ..base import ActivityClassificationResult, ACTIVITY_PROMPTS

logger = logging.getLogger("Analyzer")


class ActivityClassifier(ABC):
    """Activity classification module"""
    
    def __init__(self, model_path: str, 
                device: str = "cpu",
                confidence_threshold: Optional[float] = None):

        self.device = device   
        self.confidence_threshold = confidence_threshold
        self.classifier =  self.load_model(model_path)

        # Activity type mapping
        self.activity_prompts = ACTIVITY_PROMPTS
        self.prompt_to_activity = {v: k for k, v in self.activity_prompts.items()}

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
        vlm_model_name: str,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.7,
        model_type: Literal["chat", "text"] = "chat",
        cache: bool = True,
        llm_api_base: Optional[str] = None,
        vlm_api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.vlm_model_name = vlm_model_name
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.model_type = model_type
        self.cache = cache
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-no-key-required")
        self.vlm_api_base = self.load_api_base(vlm_api_base,"vlm")
        if self.llm_model_name:
            self.llm_api_base = self.load_api_base(llm_api_base,"llm")
        else:
            self.llm_api_base = self.vlm_api_base
    

    def load_api_base(self,api_base: Optional[str] = None,name: str = "llm") -> str:
        assert name in ["llm", "vlm"], "name must be 'llm' or 'vlm'"
        port = os.environ.get(f'{name.upper()}_PORT')
        host = os.environ.get('HOST')
        if api_base is not None:
            return api_base
        elif host and port:
            return f"http://{host}:{port}/v1"
        else:
            raise ValueError("HOST and PORT must be set in environment variables or 'api_base' should be provided as an argument")
    
    def validate_model_name(self,name: Optional[str]) -> Optional[str]:
        if str(name) == "None" or len(str(name)) == 0:
            return None
        if name.startswith("openai/"):
            pass
        elif name.startswith("ggml-org/"):
            name = "openai/" + name
        else:
            raise ValueError(f"Only OpenAI or ggml-org compatible endpoints are supported. Received: {name}")
        
        return name
    
    def validate(self) -> None:
        """Validate the model configuration."""
        self.llm_model_name = self.validate_model_name(self.llm_model_name)
        self.vlm_model_name = self.validate_model_name(self.vlm_model_name)
        
        if not (0 <= self.temperature <= 1):
            raise ValueError("Temperature must be between 0 and 2")
        
        if self.model_type not in ["chat", "text"]:
            raise ValueError("Model type must be 'chat' or 'text'")