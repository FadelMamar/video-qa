"""
Video reporting and summarization module.

This module provides functionality for generating comprehensive summaries
of video analysis results using vision-language models.
"""

import logging
from typing import List

import dspy

from ..analyzer.base import ModelConfig
from ..analyzer.dspy_analyzer import DSPyModelLoader

logger = logging.getLogger(__name__)


class VideoSummarySignature(dspy.Signature):
    """
    Analyze the descriptions of videosurveillance images and generate a comprehensive summary that includes all relevant details,
     patterns, and observations for clear decision making. 
     The summary should capture the complete narrative of activities, temporal progression, 
     and any notable events or anomalies detected across the sequence of images.
    """
    
    frames_description: str = dspy.InputField(
        desc="Sequence of frame descriptions from video surveillance"
    )
    summary: str = dspy.OutputField(
        desc="Comprehensive summary and analysis of all activities and events"
    )

class TranslatorSignature(dspy.Signature):
    """Translate the text from english to french.
    """
    text_english: str = dspy.InputField(desc="Text to translate")
    text_french: str = dspy.OutputField(desc="Translated text")

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
        self._llm = DSPyModelLoader.load_llm_model(model_config)
        self._summarizer = dspy.ChainOfThought(VideoSummarySignature)
        self._translator = dspy.ChainOfThought(TranslatorSignature)
    
    def translate(self,text:str) -> str:
        """
        Translate the text to the target language.
        """
        with dspy.context(lm=self._llm):
            response = self._translator(text_english=text)
        return response.text_french
    
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

            descriptions = [f"{frame_analyses[i]} at {timestamps[i]:.3f}s" for i in range(len(frame_analyses))]
            descriptions = "\n".join(descriptions)
            with dspy.context(lm=self._llm):
                response = self._summarizer(
                    frames_description=descriptions, 
                )
            summary = response.summary
            summary = self.translate(summary)
            return summary
            
        except Exception as e:
            logger.error(f"Video summarization failed: {e}")
            raise RuntimeError(f"Video summarization failed: {e}") from e
