"""
Video synthesis module for analyzing and summarizing video content.

This module provides a high-level interface for video analysis using
vision-language models through the DSPy framework.
"""

import io
import logging
import traceback
from typing import Union, List, Tuple, Optional
import tempfile
import os
from tqdm import tqdm
from .base import (
    ModelConfig,
    BaseAnalyzer,
)

from ..summary.reporter import VideoSummarizer
from ..config import PredictionConfig
from ..data.video_loader import DataLoading
from ..base import Video, FramesAnalysisResult,ActivityClassificationResult
from .clip_analyzer import CLIPClassifier

logger = logging.getLogger(__name__)



class VideoAnalyzer:
    """
    A class for analyzing and synthesizing video content using VLM models.
    
    This class provides a structured approach to video analysis by:
    1. Loading and processing video frames
    2. Analyzing individual frames using VLM models
    3. Generating comprehensive summaries of the analysis
    """
    
    def __init__(self, config: PredictionConfig):
        """
        Initialize the VideoSynthesizer with configuration.
        
        Args:
            config: Configuration object containing model and processing parameters
        """
        self.config = config
        self._validate_config()
        
        # Create model configuration
        self._model_config = self._create_model_config()
        
        # Initialize analyzers
        self._frame_analyzer = self._create_frame_analyzer()
        self._summarizer = self._create_summarizer()
        self._data_loader = DataLoading()
        self._clip_classifier = CLIPClassifier(
            model_path=self.config.clip_model,
            device=self.config.clip_device,
            input_size=self.config.clip_input_size
        )
        
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        #if not self.config.vlm_model:
        #    raise ValueError("Model must be specified in configuration")
        
        if self.config.temperature < 0 or self.config.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        if self.config.sample_freq <= 0:
            raise ValueError("Sample frequency must be positive")
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration from prediction config."""
        return ModelConfig(
            vlm_model_name=self.config.vlm_model,
            llm_model_name=self.config.llm_model,
            temperature=self.config.temperature,
            model_type="chat",
            cache=True,
            vlm_api_base=self.config.vlm_api_base,
            llm_api_base=self.config.llm_api_base,
            api_key=self.config.api_key
        )
    
    def _create_frame_analyzer(self) -> BaseAnalyzer:
        """Create and configure the frame analyzer based on analyzer type."""
        analyzer_type = getattr(self.config, 'analyzer_type', 'dspy').lower()
        if self.config.vlm_model is None or len(str(self.config.vlm_model)) == 0:
            logger.info("No VLM model provided, using CLIP classifier instead")
            return None
        
        if analyzer_type == 'dspy':
            from .dspy_analyzer import DSPyFrameAnalyzer
            return DSPyFrameAnalyzer(
                model_config=self._model_config,
                prompting_mode=self.config.prompting_mode
            )
        
        elif analyzer_type == 'hf':
            from .hf_analyzer import HFFrameAnalyzer, HFModelConfig
            
            # Create HF-specific config
            hf_config = HFModelConfig(
                model_name=getattr(self.config, 'hf_model_name', 'Salesforce/blip-image-captioning-base'),
                device=getattr(self.config, 'device', 'auto'),
                max_new_tokens=getattr(self.config, 'max_new_tokens', 30),
                temperature=self.config.temperature
            )
            
            return HFFrameAnalyzer(model_config=hf_config)
        
        else:
            raise ValueError(f"Unsupported analyzer type: {analyzer_type}. Supported types: dspy, hf")
    
    def _create_summarizer(self) -> VideoSummarizer:
        """Create and configure the summarizer."""
        return VideoSummarizer(model_config=self._model_config)
    
    def analyze_video(self, video: Video, cache_dir: Optional[str] = None,activity_analysis: bool = False) -> FramesAnalysisResult:
        """
        Analyze a video and return comprehensive results.
        
        Args:
            video: Video data as file path, bytes, or BytesIO object
            
        Returns:
            AnalysisResult containing frame analyses, timestamps, and summary
            
        Raises:
            ValueError: If video format is not supported
            RuntimeError: If analysis fails
        """
        # Validate input
                
        try:
            # Load and process video frames
            loader = self._data_loader.get_loader(
                video=video,
                cache_dir=cache_dir,
                sample_freq=self.config.sample_freq,
                img_as_bytes=self.config.img_as_bytes,
                max_frames=self.config.max_frames
            )
            
            # Analyze frames
            if activity_analysis or (self.config.vlm_model is None):
                frames_analysis, timestamps = self._analyze_frames_activity(loader)
            else:
                frames_analysis, timestamps = self._analyze_frames(loader)
            
            # Generate summary
            summary = self._generate_summary(frames_analysis, timestamps)
            
            return FramesAnalysisResult(
                frames_analysis=frames_analysis,
                timestamps=timestamps,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise RuntimeError(f"Video analysis failed: {str(e)}")
    
    def _analyze_frames(self, loader) -> Tuple[List[str], List[float]]:
        """
        Analyze individual frames from the video.
        
        Args:
            loader: Video frame loader iterator
            
        Returns:
            Tuple of (frame analyses, timestamps)
        """
        frames_analysis = []
        timestamps = []
        
        for data_package in tqdm(loader,desc="Analyzing frames"):
            try:
                analysis = self._frame_analyzer.analyze(data_package["frame"])
                frames_analysis.append(analysis)
                timestamps.append(data_package["timestamp"])
            except Exception as e:
                logger.warning(
                    f"Failed to analyze frame at timestamp "
                    f"{data_package.get('timestamp', 'unknown')}: {e}"
                )
                continue
        
        if not frames_analysis:
            raise RuntimeError("No frames were successfully analyzed")
        
        return frames_analysis, timestamps
    
    def _analyze_frames_activity(self, loader) -> Tuple[List[str], List[float]]:
        """
        Analyze the activity of the frames.
        """
        frames_analysis = []
        timestamps = []

        for data_package in tqdm(loader,desc="Analyzing frames"):
            try:
                image = [data_package["frame"]]
                activity_results = self._clip_classifier(image)
                activity = str(activity_results[0].activity_type)
                activity_and_conf = f"Detected activity:{activity}, confidence:{activity_results[0].confidence}"
                frames_analysis.append(activity_and_conf)
                timestamps.append(data_package["timestamp"])
            except Exception as e:
                logger.warning(
                    f"Failed to analyze frame at timestamp "
                    f"{data_package.get('timestamp', 'unknown')}: {e}"
                )
                continue
                
        return frames_analysis, timestamps

    def _generate_summary(self, frames_analysis: List[str], timestamps: List[float]) -> str:
        """
        Generate a comprehensive summary of the frame analyses.
        
        Args:
            frames_analysis: List of frame analysis strings
            timestamps: List of corresponding timestamps
            
        Returns:
            Summary string
        """
        logger.info(f"Generating summary for {len(frames_analysis)} frames")
        try:
            return self._summarizer.summarize(frames_analysis, timestamps)
        except Exception as e:
            traceback.print_exc()
            #logger.error(f"Summary generation failed: {e}")
            raise RuntimeError(f"Summary generation failed: {str(e)}") from e
    
    def get_analysis_stats(self, result: FramesAnalysisResult) -> dict:
        """
        Get statistics about the analysis results.
        
        Args:
            result: AnalysisResult object
            
        Returns:
            Dictionary containing analysis statistics
        """
        if not result.timestamps:
            return {
                "total_frames_analyzed": 0,
                "video_duration_seconds": 0,
                "average_analysis_length": 0,
                "summary_length": len(result.summary)
            }
        
        return {
            "total_frames_analyzed": len(result.frames_analysis),
            "video_duration_seconds": max(result.timestamps) - min(result.timestamps),
            "average_analysis_length": (
                sum(len(analysis) for analysis in result.frames_analysis) / 
                len(result.frames_analysis) if result.frames_analysis else 0
            ),
            "summary_length": len(result.summary)
        }


# Backward compatibility function
def analyze_video(
    video: Union[str, bytes],
    args: PredictionConfig,
    metadata: Optional[dict] = None,
    activity_analysis: bool = False
) -> FramesAnalysisResult:
    """
    Legacy function for backward compatibility.
    
    Args:
        video: Video data
        args: Configuration parameters
        
    Returns:
        Tuple of (frames_analysis, timestamps, summary)
    """
    tmpfile_path = None

    if isinstance(video, str):
        video = Video(video_path=video,metadata=metadata)
    elif isinstance(video, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video)
            tmpfile_path = tmpfile.name
            video = Video(video_path=tmpfile_path,metadata=metadata)
    else:
        raise ValueError(f"Unsupported video type: {type(video)}")

    analyzer = VideoAnalyzer(args)
    result = analyzer.analyze_video(video,cache_dir=os.environ.get("VIDEO_PREPROCESSED_DIR"),activity_analysis=activity_analysis)
    
    if tmpfile_path:
        os.remove(tmpfile_path)

    return result