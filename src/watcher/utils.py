# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:47:30 2025

@author: FADELCO
"""

import os
import hashlib
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from decord import VideoReader, cpu

from .config import PredictionConfig


def get_video_frames(args: PredictionConfig):
    

    video = args.video

    if isinstance(video, str):
        video_hash = hashlib.md5(video.encode("utf-8")).hexdigest()
        if video.startswith("http://") or video.startswith("https://"):
            raise NotImplementedError("Supports only local files")
        
    elif isinstance(video, bytes):
        # path = base64.b64encode(video) #.decode("utf-8")
        video_hash = hashlib.md5(video).hexdigest()
    
    elif isinstance(video, BytesIO):
        video_hash = hashlib.md5(video.getvalue()).hexdigest()
    
    else:
        raise ValueError(
            "video_path must be a string (file path), bytes (file content) or io.BytesIO (in-memory file)"
        )

    if isinstance(args.cache_dir,str) or isinstance(args.cache_dir, os.PathLike):
        os.makedirs(args.cache_dir, exist_ok=True)
        frames_cache_file = os.path.join(
            args.cache_dir, f"{video_hash}_{args.sample_freq}_frames.npy"
        )
        timestamps_cache_file = os.path.join(
            args.cache_dir, f"{video_hash}_{args.sample_freq}_timestamps.npy"
        )
    else:
        frames_cache_file = None
        timestamps_cache_file = None

    # Return cached frames if they exist
    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return frames, timestamps

    vr = VideoReader(video, ctx=cpu(0))
    total_frames = len(vr)

    # sampling step
    step = vr.get_avg_fps()*args.sample_freq

    indices = np.arange(0, total_frames, step)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])
    timestamps = timestamps.mean(axis=1)

    # caching frames and timestamps
    if frames_cache_file and timestamps_cache_file:
        np.save(frames_cache_file, frames)
        np.save(timestamps_cache_file, timestamps)

    return frames, timestamps


def numpy_to_bytes(image_np: np.ndarray,save_as:str="jpeg"):
    assert isinstance(image_np, np.ndarray), (
        f"Expected type 'np.ndarray' but received {type(image_np)}"
    )
    buffer = BytesIO()
    pil_image = Image.fromarray(image_np)
    pil_image.save(buffer, format=save_as)

    return buffer.getvalue()


def frame_loader(args: PredictionConfig, img_as_bytes: bool = True):
    frames, timestamps = get_video_frames(args)

    func = lambda x: x
    if img_as_bytes:
        func = lambda x: numpy_to_bytes(x, save_as=args.save_as)

    num_frames = len(frames)
    batch = args.batch_frames
    for i in range(0, num_frames, batch):
        img_batch = [func(frames[i+k]) for k in range(batch) if i + k < num_frames]
        yield img_batch, timestamps[i : min(num_frames, i + batch)]




