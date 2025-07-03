"""
CLIP-based Zero-Shot Activity Classifier

This module provides activity recognition using CLIP (Contrastive Language-Image Pre-training)
models for zero-shot classification. It inherits from the base ActivityClassifier and uses
CLIP's ability to classify images based on text descriptions without fine-tuning.
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

from transformers import CLIPProcessor, CLIPModel

from .base import ActivityClassifier, ActivityClassificationResult, ActivityType, logger

class CLIPClassifier(ActivityClassifier):
    """
    CLIP-based zero-shot activity classifier
    
    Uses CLIP models to perform zero-shot activity classification based on
    text descriptions. Supports various CLIP variants like SigLIP, OpenCLIP,
    and other contrastive vision-language models.
    """
    
    def __init__(
        self,
        model_path: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        model_type: str = "auto",  # "auto", "clip", "siglip", "openclip",
        input_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize CLIP activity classifier
        
        Args:
            model_path: Path to CLIP model or model identifier
            device: Device to use ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence for activity classification
            ambiguous_threshold: Threshold below which activities are marked as ambiguous
            model_type: Type of CLIP model ("auto", "clip", "siglip", "openclip", "custom")
            input_size: Input image size for the model
        """
        
        self.model_type = model_type
        self.input_size = input_size
        
        # Set default activity prompts if not provided
        
        
        # Initialize base class
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        
        logger.info(f"Initialized CLIP classifier with model: {model_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load the CLIP model from Hugging Face
        
        Args:
            model_path: Path to model or model identifier
            
        Returns:
            Loaded model and processor
        """
        logger.info(f"Loading CLIP model from: {model_path}")
        
        # Auto-detect model type if not specified
        if self.model_type == "auto":
            self.model_type = self._detect_model_type(model_path)
        
        return self._load_clip_variant(model_path, model_type=self.model_type)
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type based on model name"""
        model_path_lower = model_path.lower()
        
        if "siglip" in model_path_lower:
            return "siglip"
        elif "openclip" in model_path_lower:
            return "openclip"
        elif "clip" in model_path_lower:
            return "clip"
        else:
            return "generic"

    def _load_clip_variant(self, model_path: str, model_type: str) -> Dict[str, Any]:
        """
        Helper to load any CLIP-compatible model with processor and device handling.
        """
        try:
            processor = CLIPProcessor.from_pretrained(model_path)
            model = CLIPModel.from_pretrained(model_path).to(self.device)

            # Move model to device
            if self.device != "cpu":
                try:
                    model = model.half()  # Use half precision for GPU
                except Exception as e:
                    logger.error(f"Failed to convert model to half precision: {e}")
                    pass

            model.eval()
            return {"model": model, "processor": processor, "type": model_type}
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise

    def preprocess(self, images: List[np.ndarray]) -> List[Image.Image]:
        """
        Preprocess images for CLIP input
        
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
            resized_img = cv2.resize(rgb_img, self.input_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(resized_img)
            processed_images.append(pil_img)
        
        return processed_images
        
    def _classify_single_image(self, image: Image.Image) -> Tuple[float, ActivityType]:
        """
        Classify a single image using zero-shot CLIP classification
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Tuple of (confidence, ActivityType)
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        model = self.classifier["model"]
        processor = self.classifier["processor"]
        
        # Generate activity labels
        activity_indices = [self.activity_to_idx[act] for act in self.activity_prompts.keys()]
        prompts = [self.activity_prompts[self.idx_to_activity[idx]] for idx in activity_indices]

        
        # Prepare inputs for CLIP
        inputs = processor(
            text=prompts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1).squeeze()
        
        # Convert to numpy for easier handling
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        probs = probs.cpu().numpy()
        
        # Find the most likely activity
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        
        # Map index to activity type
        predicted_activity = self.idx_to_activity[max_idx]
        
        return max_prob, predicted_activity
    
    def __call__(self, images: List[np.ndarray]) -> List[ActivityClassificationResult]:
        """
        Classify activity for a tracked object using zero-shot CLIP classification
        
        Args:
            images: List of images (numpy arrays) to analyze
            obj: TrackedObject containing tracking information
            
        Returns:
            ActivityResult with classification results
        """
        if not images:
            raise ValueError("No images provided")
        
        # Preprocess images
        processed_images = self.preprocess(images)
        
        # Classify each image
        frame_results = []
        for img in processed_images:
            confidence, activity_type = self._classify_single_image(img)
            
            # Create result
            result = ActivityClassificationResult(
                activity_type=activity_type,
                confidence=confidence,
            )

            frame_results.append(result)
        
        return frame_results
    
            
def create_clip_classifier(
    model_path: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    confidence_threshold: float = 0.5,
) -> CLIPClassifier:
    """
    Factory function to create a CLIP classifier
    
    Args:
        model_path: CLIP model path or identifier
        device: Device to use
        confidence_threshold: Minimum confidence for classification
        ambiguous_threshold: Threshold for ambiguous results
        **kwargs: Additional arguments for CLIPClassifier
        
    Returns:
        Configured CLIPClassifier instance
    """
    return CLIPClassifier(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
