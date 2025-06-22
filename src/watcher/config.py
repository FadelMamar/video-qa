# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:25:14 2025

@author: FADELCO
"""

from dataclasses import dataclass
import io


@dataclass
class PredictionConfig:
    video: str|bytes|io.BytesIO = None

    model:str=None
    temperature: float = 0.7
    prompting_mode:str="basic"  # Options: "basic", "cot"

    sample_freq: int = 5
    batch_frames: int = 5

    cache_dir: str = ".cache"
    save_as: str = "jpeg"  # Format to save frames, e.g., "jpeg", "png"
    

    def __post_init__(self):
        # if not isinstance(self.video,str) or not isinstance(self.video,io.BytesIO):
        #     raise ValueError(f"video is of type 'str' or 'io.BytesIO' only. Received {type(self.video)}")
        if not isinstance(self.sample_freq, int) or self.sample_freq <= 0:
            raise ValueError("sample_freq must be a positive integer")
        if not isinstance(self.batch_frames, int) or self.batch_frames <= 0:
            raise ValueError("batch_frames must be a positive integer")
        if not self.cache_dir:
            self.cache_dir = ".cache"
