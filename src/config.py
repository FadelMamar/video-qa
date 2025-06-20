# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:25:14 2025

@author: FADELCO
"""

from dataclasses import dataclass

@dataclass
class PredictionConfig:
    
    video_path:str = None
    sample_freq:int=5
    cache_dir:str=".cache"
    
    batch_frames:int = 5