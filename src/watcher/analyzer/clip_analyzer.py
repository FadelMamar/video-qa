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
from cv2.gapi import imgproc
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import json
from pathlib import Path
from io import BytesIO

from transformers import pipeline

from .base import ActivityClassifier, ActivityClassificationResult, logger
from ..base import ActivityType

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
        
        self.input_size = input_size      
        
        # Initialize base class
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=None,
        )
        
        logger.info(f"Initialized CLIP classifier with model: {model_path}")
    
    def load_model(self, model_path: str) -> pipeline:
        """
        Load the CLIP model from Hugging Face
        
        Args:
            model_path: Path to model or model identifier
            
        Returns:
            Loaded model and processor
        """
        logger.info(f"Loading CLIP model from: {model_path}")
        try:
            return pipeline(model=model_path, task="zero-shot-image-classification",device=self.device,)
        except Exception as e:
            logger.error(f"Failed to load {model_path} model: {e}")
            raise
    
    def preprocess(self, images: List[bytes]) -> List[Image.Image]:
        """
        Preprocess images for CLIP input
        
        Args:
            images: List of numpy arrays (BGR format from OpenCV)
            
        Returns:
            List of PIL Images (RGB format)
        """
        processed_images = []
        
        for img in images:
            assert isinstance(img, bytes), "Image must be bytes, received: " + str(type(img))

            #buff = BytesIO(img)
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to input size
            resized_img = cv2.resize(rgb_img, self.input_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(resized_img)
            processed_images.append(pil_img)
        
        return processed_images
    
    def postprocess(self, activity_type: ActivityType, confidence: float) -> ActivityClassificationResult:
        # Create result
        return ActivityClassificationResult(
            activity_type=activity_type,
            confidence=confidence,
        )

    def _classify_batch(self, images: List[Image.Image]) -> Tuple[float, ActivityType]:
        """
        Classify a single image using zero-shot CLIP classification
        
        Args:
            images: List of PIL Images to classify
            
        Returns:
            Tuple of (confidence, ActivityType)
        """
        if self.classifier is None:
            raise ValueError("Classifier not loaded")
        
        assert isinstance(images, list), "Images must be a list, received: " + str(type(images))
                
        # Generate activity labels
        prompts = list(self.activity_prompts.values())
        
        def get_indx(result: List[Dict[str, float]]) -> int:
            scores = [o['score'] for o in result]
            scores = np.array(scores) / np.sum(scores)
            idx = int(np.argmax(scores))
            label = result[idx]['label']
            score = float(round(scores[idx],3))
            return idx,score,label

        # Get model predictions
        with torch.no_grad():
            outputs = self.classifier(images,candidate_labels=prompts)
        indices,probs,labels = zip(*[get_indx(result) for result in outputs])
        predicted_activity = [self.prompt_to_activity[label] for label in labels]

        #print(json.dumps(outputs, indent=4))
        #print(probs,labels)
                
        return probs, predicted_activity
    
    def __call__(self, images: List[bytes]) -> List[ActivityClassificationResult]:
        """
        Classify activity for a tracked object using zero-shot CLIP classification
        
        Args:
            images: List of images (numpy arrays) to analyze
            obj: TrackedObject containing tracking information
            
        Returns:
            ActivityResult with classification results
        """
        
        # Preprocess images
        assert isinstance(images, list), "Images must be a list, received: " + str(type(images))
        processed_images = self.preprocess(images)
        confidences, activity_types = self._classify_batch(processed_images)

        # Classify each image
        frame_results = []
        for activity_type, conf in zip(activity_types, confidences):
            result = self.postprocess(activity_type=activity_type, confidence=conf)
            frame_results.append(result)
        
        return frame_results
    
            
def create_clip_classifier(
    model_path: str = "openai/clip-vit-base-patch32",
    device: str = "auto",
    input_size: Tuple[int, int] = (1024, 1024),
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
        input_size=input_size,
    )
