"""
Visual Language Model (VLM) Activity Classifier

This module provides activity recognition using visual-language models from Hugging Face.
It inherits from the base ActivityClassifier and uses transformer-based VLMs to analyze
image sequences and classify activities based on visual content and language prompts.
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import json
from pathlib import Path

import dspy
import spacy

from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    AutoModel,
    pipeline
)

import io

from .base import ActivityClassifier, ActivityResult, TrackedObject, ActivityType, ACTIVITY_PROMPTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivityClassificationSignature(dspy.Signature):
    """Analyze the image and classify the activity being performed.
       You should accurately identify what activity is happening in the image.
       
       Look for these specific activities:
       - Normal walking: A person walking normally
       - Running: A person running or jogging  
       - Fighting: People fighting or engaging in physical conflict
       - Group gathering: A group of people gathering together
       - Militia behavior: People in military or militia formation
       - Vehicle driving: A vehicle driving on the road
       - Vehicle speeding: A vehicle driving at high speed
       - Vehicle stopping: A vehicle stopping or parked
       - Suspicious behavior: Suspicious or unusual behavior
       - Crowd formation: A large crowd forming or moving
       - Unknown: Unknown or unclear activity
    """
    
    image: dspy.Image = dspy.InputField(
        desc="image from video surveillance showing an activity"
    )
    activity_prompt: str = dspy.InputField(
        desc="specific prompt describing the activity to look for"
    )
    activity_description: str = dspy.OutputField(
        desc="detailed description of what activity is happening in the image"
    )


class DspyActivityClassifier:
    """
    DSPy-based activity classifier for OpenAI-compatible endpoints
    
    Wraps the logic for sending requests to OpenAI endpoints using DSPy,
    providing a clean interface for activity classification.
    """
    
    def __init__(
        self,
        model: str = "openai/Qwen2.5-VL-3B",
        temperature: float = 0.7,
        prompting_mode: str = "basic",
        cache: bool = True,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize DSPy activity classifier
        
        Args:
            model: Model identifier (should start with "openai/")
            temperature: Sampling temperature for generation
            prompting_mode: "basic" or "cot" (chain of thought)
            cache: Whether to cache responses
            api_base: Custom API base URL
            api_key: Custom API key
        """
                
                
        assert prompting_mode in ["basic", "cot"], f"{prompting_mode} not supported."
        
        # Set environment variables for DSPy
        if api_base:
            os.environ["BASE_URL"] = api_base
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Load DSPy model
        self.lm = self._load_dspy_model(
            model=model, 
            temperature=temperature, 
            cache=cache
        )
        
        # Configure DSPy predictor
        if prompting_mode == "cot":
            self.predictor = dspy.ChainOfThought(ActivityClassificationSignature)
        else:
            self.predictor = dspy.Predict(ActivityClassificationSignature)
        
        logger.info(f"Initialized DSPy activity classifier with model: {model}")
    
    def _load_dspy_model(
        self,
        model: str, 
        temperature: float, 
        cache: bool = True
    ):
        """Load DSPy language model"""
        lm = dspy.LM(
            model,
            api_key=os.environ.get("OPENAI_API_KEY", "sk-no-key-required"),
            temperature=temperature,
            api_base=os.environ.get("BASE_URL"),
            model_type="chat",
            cache=cache,
        )
        return lm
    
    def classify_activity(self, image: Image.Image, activity_prompt: str) -> str:
        """
        Classify activity in an image using DSPy
        
        Args:
            image: PIL Image to analyze
            activity_prompt: Prompt describing the activity to look for
            
        Returns:
            Generated description of th
            e activity
        """
        # Convert PIL image to bytes for DSPy
        assert isinstance(image, Image.Image), "Image must be a PIL Image"
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue() # type: ignore
        
        # Create DSPy image
        dspy_image = dspy.Image.from_file(img_bytes)
        
        # Run prediction with DSPy context
        with dspy.context(lm=self.lm):
            response = self.predictor(
                image=dspy_image,
                activity_prompt=activity_prompt
            )
        
        return response.activity_description
            
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the DSPy model configuration"""
        return {
            "model": getattr(self.lm, 'model', 'unknown'),
            "temperature": getattr(self.lm, 'temperature', 0.7),
            "api_base": os.environ.get("BASE_URL", "default"),
            "prompting_mode": "cot" if isinstance(self.predictor, dspy.ChainOfThought) else "basic"
        }


class VLMClassifier(ActivityClassifier):
    """
    Visual Language Model-based activity classifier
    
    Uses Hugging Face transformers to analyze image sequences and classify activities
    based on visual content and language prompts. Supports various VLM architectures
    like BLIP, LLaVA, and other vision-language models.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        ambiguous_threshold: float = 0.3,
        model_type: str = "auto",  # "auto", "blip", "llava", "custom"
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        input_size: Tuple[int, int] = (224, 224),
        use_temporal_aggregation: bool = True,
    ):
        """
        Initialize VLM activity classifier
        
        Args:
            model_path: Path to Hugging Face model or model identifier
            device: Device to use ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence for activity classification
            ambiguous_threshold: Threshold below which activities are marked as ambiguous
            model_type: Type of VLM model ("auto", "blip", "llava", "custom")
            max_length: Maximum sequence length for text generation
            temperature: Sampling temperature for text generation
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling for text generation
            num_beams: Number of beams for beam search
            input_size: Input image size for the model
        """
        
        self.model_type = model_type
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.input_size = input_size

        self._nlp = spacy.load("en_core_web_sm")

        self.torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        
        # Set default activity prompts if not provided
        self.activity_prompts = ACTIVITY_PROMPTS
        
        # Initialize base class
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            ambiguous_threshold=ambiguous_threshold
        )
        
        logger.info(f"Initialized VLM classifier with model: {model_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load the VLM model from Hugging Face
        
        Args:
            model_path: Path to model or model identifier
            
        Returns:
            Loaded model and processor
        """
        logger.info(f"Loading VLM model from: {model_path}")
        
        # Auto-detect model type if not specified
        if self.model_type == "auto":
            self.model_type = self._detect_model_type(model_path)
        
        # Load model based on type
        if self.model_type == "blip":
            return self._load_model(model_path, "blip")

        elif self.model_type == "llava":
            return self._load_model(model_path, "llava")

        elif self.model_type == "custom":
            return self._load_custom_model(model_path)

        elif self.model_type == "dspy":
            return self._load_dspy_model(model_path)
        else:
                # Try generic loading
                return self._load_generic_model(model_path)
                
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type based on model name"""
        model_path_lower = model_path.lower()
        
        if "blip" in model_path_lower:
            return "blip"
        elif "llava" in model_path_lower:
            return "llava"
        elif "qwen" in model_path_lower and "vl" in model_path_lower:
            return "custom"
        elif model_path.startswith("http") or "api" in model_path_lower:
            return "openai_api"
        elif model_path.startswith("openai/"):
            return "dspy"
        else:
            return "generic"
    
    def _load_model(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """Load VLM model based on type"""
        processor = AutoProcessor.from_pretrained(model_path)

        model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            ).eval()
        
        if self.device == "cpu":
            model = model.to(self.device)
        
        return {"model": model, "processor": processor, "type": model_type}
    
    def _load_custom_model(self, model_path: str) -> Dict[str, Any]:
        """Load custom VLM model (e.g., Qwen-VL)"""
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModel(
                model_path,
                device=self.device,
                torch_dtype=self.torch_dtype
            )
            return {"model": model,"processor": processor, "type": "custom"}
        except:
            # Fallback to generic loading
            return self._load_generic_model(model_path)
    
    def _load_generic_model(self, model_path: str) -> Dict[str, Any]:
        """Load generic VLM model"""
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            ).eval()
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            return {"model": model, "processor": processor, "type": "generic"}
        except Exception as e:
            logger.error(f"Failed to load model generically: {e}")
            raise
        
    def _load_dspy_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load DSPy-based model for activity classification
        
        Args:
            model_path: DSPy model identifier (should start with "openai/")
            
        Returns:
            Dictionary containing DSPy classifier
        """
        
        # Create DSPy activity classifier
        dspy_classifier = DspyActivityClassifier(
            model=model_path,
            temperature=self.temperature,
            prompting_mode="basic",  # Can be made configurable
            cache=True,
            api_base=os.environ.get("BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        return {
            "classifier": dspy_classifier,
            "type": "dspy",
            "processor": None
        }

    def preprocess(self, images: List[np.ndarray]) -> List[Image.Image]:
        """
        Preprocess images for VLM input
        
        Args:
            images: List of numpy arrays (BGR format from OpenCV)
            
        Returns:
            List of PIL Images (RGB format)
        """
        processed_images = []
        
        for img in images:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to input size
            resized_img = cv2.resize(rgb_img, self.input_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(resized_img)
            processed_images.append(pil_img)
        
        return processed_images
    
    def _generate_activity_prompt(self,) -> str:
        """Generate prompt for activity classification"""
        # Get all possible activities from prompts
        all_activities = list(self.activity_prompts.values())
        activities_text = ", ".join(all_activities)
        return f"Describe what activity is happening in this image. Look for these activities: {activities_text}."
    
    def _classify_single_image(self, image: Image.Image,) -> Tuple[float, ActivityType]:
        """
        Classify a single image
        
        Args:
            image: PIL Image to classify
            activity_type: Activity type to check for
            
        Returns:
            Confidence score for the activity
        """
        prompt = self._generate_activity_prompt()
        
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        elif self.classifier["type"] == "dspy":
            # Use DSPy approach
            dspy_classifier = self.classifier["classifier"]
            predicted_activity = dspy_classifier.classify_activity(image, prompt)
        else:
            # Use model directly
            model = self.classifier["model"]
            processor = self.classifier["processor"]
            
            # Prepare inputs
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                predicted_activity = model.generate(**inputs)
                        
        # Calculate confidence based on text similarity
        confidence, best_activity = self._calculate_activity_confidence(predicted_activity)

        return confidence, best_activity
        
    
    def _calculate_activity_confidence(self, generated_text: str) -> float:
        """
        Calculate confidence score based on generated text using spaCy similarity
        
        Args:
            generated_text: Text generated by the VLM
            
        Returns:
            Confidence score between 0 and 1
        """
        
        # Load spaCy model (cache it for performance)
        if not hasattr(self, '_nlp'):
            self._nlp = spacy.load("en_core_web_sm")
        
        # Define activity keywords for all available activities
        activity_keywords = {
            ActivityType.NORMAL_WALKING: ["walking", "walk", "strolling", "ambling"],
            ActivityType.RUNNING: ["running", "run", "jogging", "sprinting"],
            ActivityType.FIGHTING: ["fighting", "fight", "conflict", "combat", "brawl"],
            ActivityType.GROUP_GATHERING: ["gathering", "group", "crowd", "assembly", "meeting"],
            ActivityType.MILITIA_BEHAVIOR: ["military", "militia", "formation", "drill", "training"],
            ActivityType.VEHICLE_DRIVING: ["driving", "vehicle", "car", "truck", "moving"],
            ActivityType.VEHICLE_SPEEDING: ["speeding", "fast", "rapid", "quick", "high speed"],
            ActivityType.VEHICLE_STOPPING: ["stopping", "stopped", "parked", "stationary"],
            ActivityType.SUSPICIOUS_BEHAVIOR: ["suspicious", "unusual", "strange", "odd"],
            ActivityType.CROWD_FORMATION: ["crowd", "large group", "mass", "formation"],
            ActivityType.UNKNOWN: ["unknown", "unclear", "unidentifiable"]
        }
        
        # Process generated text with spaCy
        doc_generated = self._nlp(generated_text.lower())
        
        # Calculate similarity with each activity type
        max_similarity = 0.0
        best_activity = None
        
        for activity_type, keywords in activity_keywords.items():
            # Create activity description from keywords
            activity_description = " ".join(keywords)
            doc_activity = self._nlp(activity_description.lower())
            
            # Calculate similarity
            similarity = doc_generated.similarity(doc_activity)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_activity = activity_type
        
        # Apply keyword-based adjustments
        text_lower = generated_text.lower()
        
        # Count keyword matches for the best activity
        if best_activity:
            keywords = activity_keywords[best_activity]
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_bonus = min(0.2, matches * 0.05)  # Max 0.2 bonus
        else:
            keyword_bonus = 0.0
        
        # Check for negative indicators
        negative_keywords = ["not", "no", "none", "absent", "empty", "nothing"]
        negative_matches = sum(1 for keyword in negative_keywords if keyword in text_lower)
        negative_penalty = min(0.3, negative_matches * 0.1)  # Max 0.3 penalty
        
        # Combine similarity with keyword adjustments
        confidence = max(0.0, min(1.0, max_similarity + keyword_bonus - negative_penalty))
        
        return confidence, best_activity
            
        
    def __call__(self, images: List[np.ndarray], obj: TrackedObject) -> ActivityResult:
        """
        Classify activity for a tracked object
        
        Args:
            images: List of images (numpy arrays) to analyze
            obj: TrackedObject containing tracking information
            
        Returns:
            ActivityResult with classification results
        """
        if not images:
            return self._create_unknown_result(obj)
        
        # Preprocess images
        processed_images = self.preprocess(images)
        
        results = []
        # Classify each activity type
        for img in processed_images:
            confidence, activity_type = self._classify_single_image(img)
         
            # Apply confidence thresholds
            if confidence < self.ambiguous_threshold:
                activity_type = ActivityType.UNKNOWN
                supporting_evidence = {
                    "ambiguous": True, 
                    "raw_confidence": confidence,
                }
            elif confidence < self.confidence_threshold:
                supporting_evidence = {
                    "ambiguous": True, 
                    "raw_confidence": confidence,
                }
            else:
                supporting_evidence = {
                    "confident": True, 
                    "raw_confidence": confidence,
                }
        
            # Create result
            result = ActivityResult(
                track_id=obj.track_id,
                activity_type=activity_type,
                confidence=confidence,
                start_timestamp=obj.timestamps[0],
                end_timestamp=obj.timestamps[-1],
                bounding_region=self._calculate_bounding_region(obj.bboxes),
                supporting_evidence=supporting_evidence,
                object_class=obj.class_name,
                frame_sequence_length=len(processed_images)
            )
            results.append(result)

        return results
    
    def _create_unknown_result(self, obj: TrackedObject) -> ActivityResult:
        """Create unknown activity result when no images are available"""
        return ActivityResult(
            track_id=obj.track_id,
            activity_type=ActivityType.UNKNOWN,
            confidence=0.0,
            start_timestamp=obj.timestamps[0] if obj.timestamps else 0.0,
            end_timestamp=obj.timestamps[-1] if obj.timestamps else 0.0,
            bounding_region=self._calculate_bounding_region(obj.bboxes),
            supporting_evidence={"no_images": True},
            object_class=obj.class_name,
            frame_sequence_length=0
        )
    
    def _calculate_bounding_region(self, bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """Calculate overall bounding region from sequence of bounding boxes"""
        if not bboxes:
            return (0, 0, 0, 0)
        
        x1_min = min(bbox[0] for bbox in bboxes)
        y1_min = min(bbox[1] for bbox in bboxes)
        x2_max = max(bbox[2] for bbox in bboxes)
        y2_max = max(bbox[3] for bbox in bboxes)
        
        return (x1_min, y1_min, x2_max, y2_max)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "model_path": getattr(self, 'model_path', 'unknown'),
            "device": self.device,
            "input_size": self.input_size,
            "activity_prompts": {k.value: v for k, v in self.activity_prompts.items()}
        }


# Example usage:
# 
# # For Hugging Face models:
# classifier = VLMClassifier(
#     model_path="Salesforce/blip2-opt-2.7b",
#     model_type="blip",
#     device="cuda"
# )
#
#
# # For DSPy-based classification:
# classifier = VLMClassifier(
#     model_path="openai/Qwen2.5-VL-3B",
#     model_type="dspy",
#     device="cpu"
# )
#

