# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:25:14 2025

@author: FADELCO
"""

from dataclasses import dataclass
import io
from typing import Optional, Union, Sequence


@dataclass
class PredictionConfig:

    vlm_model: Optional[str] = None
    temperature: float = 0.7
    prompting_mode:str="basic"  # Options: "basic", "cot"

    clip_model:str="google/siglip2-base-patch16-224"
    clip_device:str="cpu"
    clip_input_size:Sequence[int]=(1024,1024)

    img_as_bytes:bool=True

    sample_freq: int = 5
    batch_frames: int = 1
    max_frames: Optional[int] = None  # Maximum frames to extract (None for all)

    save_as: str = "jpeg"  # Format to save frames, e.g., "jpeg", "png"
    image_quality: int = 85  # JPEG quality (1-100, only for JPEG)
    

    def __post_init__(self):
        # if not isinstance(self.video,str) or not isinstance(self.video,io.BytesIO):
        #     raise ValueError(f"video is of type 'str' or 'io.BytesIO' only. Received {type(self.video)}")
        if not isinstance(self.sample_freq, int) or self.sample_freq <= 0:
            raise ValueError("sample_freq must be a positive integer")
        if not isinstance(self.batch_frames, int) or self.batch_frames <= 0:
            raise ValueError("batch_frames must be a positive integer")
        
        if not isinstance(self.image_quality, int) or self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("image_quality must be an integer between 1 and 100")
        
        assert self.max_frames is None or self.max_frames > 0, "max_frames must be a positive integer"
