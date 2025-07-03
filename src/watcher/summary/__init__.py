"""
Video Summary Module

This module provides functionality for generating comprehensive summaries
of video analysis results using vision-language models.
"""

from .reporter import VideoSummarizer, VideoSummarySignature

__all__ = [
    "VideoSummarizer",
    "VideoSummarySignature",
]

__version__ = "1.0.0"
