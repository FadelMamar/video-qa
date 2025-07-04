"""
Vision Language Model (VLM) integration module for video analysis.

This module provides classes and utilities for analyzing video frames using
vision-language models through the DSPy framework.
"""

import json
from typing import List, Literal, Union, Optional, Dict, Any

import dspy

from .base import (
    ModelConfig,
    BaseAnalyzer,
    logger
)

from ..base import ACTIVITY_PROMPTS, ActivityType

LIST_OF_ACTIVITIES = list(ACTIVITY_PROMPTS.values())

class FrameAnalysisSignature(dspy.Signature):
    """
    Analzye the sequence of images from aerial video surveillance footage and accuractly identify and classify any activity. 
    You should describe accurately what is happening.
    For example, if a person is walking, running, or standing still, if there is a vehicle, a motorcycle doing any activity.

    """
    
    images: List[dspy.Image] = dspy.InputField(
        desc="Sequence of images from video surveillance camera"
    )
    analysis: str = dspy.OutputField(
        desc="Detailed analysis of the sequence of aerial images"
    )


class ActivityAnalysisSignature(dspy.Signature):
    """
    Analyze the image sequence and identify specific activities being performed.
    Focus on security-relevant behaviors and activities.
    """
    
    images: dspy.Image = dspy.InputField(
        desc="Sequence of images showing tracked object activity"
    )
    list_of_activities: List[str] = dspy.InputField(
        desc="List of activities to look for."
    )
    context: str = dspy.InputField(
        desc="Additional context about the tracked object"
    )
    detected_activity: str = dspy.OutputField(
        desc="detected activity in the image"
    )
    confidence_score: float = dspy.OutputField(
        desc="Confidence score for the activity (0.0 to 1.0)"
    )


class DSPyModelLoader:
    """Utility class for loading and configuring DSPy models."""
    
    @staticmethod
    def load_vlm_model(config: ModelConfig) -> dspy.LM:
        """
        Load a DSPy language model with the given configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Configured DSPy language model
        """
        config.validate()
        
        return dspy.LM(
            config.vlm_model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            api_base=config.vlm_api_base,
            model_type=config.model_type,
            cache=config.cache,
        )
    
    @staticmethod
    def load_llm_model(config: ModelConfig) -> dspy.LM:
        """
        Load a DSPy language model with the given configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Configured DSPy language model
        """
        config.validate()
        
        return dspy.LM(
            config.llm_model_name,
            api_key=config.api_key,
            temperature=config.temperature,
            api_base=config.llm_api_base,
            model_type=config.model_type,
            cache=config.cache,
        )



class DSPyFrameAnalyzer(BaseAnalyzer):
    """
    DSPy-based frame analyzer for video surveillance.
    
    Uses DSPy framework to analyze sequences of video frames and identify
    activities, objects, and events in surveillance footage.
    """
    
    PROMPTING_MODES = {"basic", "cot"}
    
    def __init__(
        self,
        model_config: ModelConfig,
        prompting_mode: str = "basic",
    ):
        """
        Initialize the DSPy frame analyzer.
        
        Args:
            model_config: Configuration for the VLM model
            prompting_mode: Prompting strategy ("basic" or "cot")
            
        Raises:
            ValueError: If prompting mode is invalid
        """
        if prompting_mode not in self.PROMPTING_MODES:
            raise ValueError(f"Prompting mode must be one of {self.PROMPTING_MODES}")
        
        self.model_config = model_config
        self.prompting_mode = prompting_mode
        
        # Initialize DSPy components
        self._vlm = DSPyModelLoader.load_vlm_model(model_config)
        self._predictor = self._create_predictor()
        self._activity_predictor = self._create_activity_predictor()
    
    def _create_predictor(self) -> Union[dspy.Predict, dspy.ChainOfThought]:
        """Create the appropriate DSPy predictor based on prompting mode."""
        if self.prompting_mode == "cot":
            return dspy.ChainOfThought(FrameAnalysisSignature)
        return dspy.Predict(FrameAnalysisSignature)
    
    def _create_activity_predictor(self) -> Union[dspy.Predict, dspy.ChainOfThought]:
        """Create the appropriate DSPy predictor for activity analysis."""
        if self.prompting_mode == "cot":
            return dspy.ChainOfThought(ActivityAnalysisSignature)
        return dspy.Predict(ActivityAnalysisSignature)
    
    def analyze(self, images: Union[bytes, List[bytes]]) -> str:
        """
        Analyze a sequence of images using DSPy.
        
        Args:
            images: Single image bytes or list of image bytes
            
        Returns:
            Analysis result as string
            
        Raises:
            RuntimeError: If analysis fails
        """

        # Validate and normalize input
        image_list = self._validate_images(images)

        dspy_images = [dspy.Image.from_file(img) for img in image_list]
        
        # Run analysis
        with dspy.context(lm=self._vlm):
            response = self._predictor(images=dspy_images)
        
        return response.analysis
    
    def analyze_activity(
        self, 
        image: bytes, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze images for specific activity recognition using DSPy.
        
        Args:
            images: Single image bytes or list of image bytes
            activity_type: Specific activity type to look for
            context: Additional context information
            
        Returns:
            Activity analysis result as string
        """
        raise NotImplementedError("Activity analysis is not implemented for DSPy")
        # Validate and normalize input
        self._validate_images(image)
        image = self.load_image_as_pil(image)
        dspy_images = dspy.Image.from_PIL(image)
        
        # Prepare activity type and context
        if context:
            context_str = json.dumps(context)
        else:
            context_str = "no additional context"
        
        # Run activity analysis
        with dspy.context(lm=self._vlm):
            response = self._activity_predictor(
                images=dspy_images,
                list_of_activities=LIST_OF_ACTIVITIES,
                context=context_str
            )
        
        return response.detected_activity, response.confidence_score
    

