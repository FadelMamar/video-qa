"""
Video reporting and summarization module.

This module provides functionality for generating comprehensive summaries
of video analysis results using vision-language models.
"""

import logging
from typing import List

import dspy

from ..agent.vlm import ModelConfig, DSPyModelLoader

logger = logging.getLogger(__name__)


class VideoSummarySignature(dspy.Signature):
    """
    Analyze the descriptions of videosurveillance images and generate a comprehensive summary that includes all relevant details,
     patterns, and observations for clear decision making. 
     The summary should capture the complete narrative of activities, temporal progression, 
     and any notable events or anomalies detected across the sequence of images.
    """
    
    frame_descriptions: List[str] = dspy.InputField(
        desc="Sequence of frame descriptions from video surveillance"
    )
    timestamps: List[float] = dspy.InputField(
        desc="Timestamps of frames in seconds since video start"
    )
    summary: str = dspy.OutputField(
        desc="Comprehensive summary of all activities and events"
    )


class VideoSummarizer:
    """
    Summarizer for video analysis results.
    
    Takes frame-by-frame analyses and generates comprehensive summaries
    of the entire video sequence.
    """
    
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the video summarizer.
        
        Args:
            model_config: Configuration for the VLM model
        """
        self.model_config = model_config
        self._lm = DSPyModelLoader.load_model(model_config)
        self._summarizer = dspy.ChainOfThought(VideoSummarySignature)
    
    def summarize(
        self, 
        frame_analyses: List[str], 
        timestamps: List[float]
    ) -> str:
        """
        Generate a comprehensive summary of frame analyses.
        
        Args:
            frame_analyses: List of frame analysis strings
            timestamps: Corresponding timestamps for each frame
            
        Returns:
            Summary string
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If summarization fails
        """
        if not frame_analyses:
            raise ValueError("No frame analyses provided")
        
        if len(frame_analyses) != len(timestamps):
            raise ValueError("Number of analyses must match number of timestamps")
        
        try:
            with dspy.context(lm=self._lm):
                response = self._summarizer(
                    frame_descriptions=frame_analyses, 
                    timestamps=timestamps
                )
            
            return response.summary
            
        except Exception as e:
            logger.error(f"Video summarization failed: {e}")
            raise RuntimeError(f"Video summarization failed: {e}") from e
