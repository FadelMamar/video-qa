# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:25:14 2025

@author: FADELCO
"""

from dataclasses import dataclass
import io
from typing import Optional, Union, Sequence
import torch

@dataclass
class PredictionConfig:

    vlm_model: Optional[str] = "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m"
    llm_model: Optional[str] = "ggml-org/Qwen3-0.6B-GGUF:f16"
    vlm_api_base: Optional[str] = None
    llm_api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    prompting_mode:str="basic"  # Options: "basic", "cot"

    analyzer_type:str="dspy"

    clip_model:str="google/siglip2-base-patch16-224"
    clip_device:str="auto"  # Options: "auto", "cpu", "cuda"
    clip_input_size:Sequence[int]=(1024,1024)

    img_as_bytes:bool=True

    sample_freq: int = 5
    batch_frames: int = 1
    max_frames: Optional[int] = None  # Maximum frames to extract (None for all)

    save_as: str = "jpeg"  # Format to save frames, e.g., "jpeg", "png"
    image_quality: int = 85  # JPEG quality (1-100, only for JPEG)
    

    def __post_init__(self):
        
        if not isinstance(self.sample_freq, int) or self.sample_freq <= 0:
            raise ValueError("sample_freq must be a positive integer")
        if not isinstance(self.batch_frames, int) or self.batch_frames <= 0:
            raise ValueError("batch_frames must be a positive integer")
        
        if not isinstance(self.image_quality, int) or self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("image_quality must be an integer between 1 and 100")
        
        assert self.max_frames is None or self.max_frames > 0, "max_frames must be a positive integer"

