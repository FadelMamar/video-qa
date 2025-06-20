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
import decord
from decord import VideoReader, cpu

from config import PredictionConfig

def get_video_frames(args:PredictionConfig):
    
    os.makedirs(args.cache_dir, exist_ok=True)

    video_hash = hashlib.md5(args.video_path.encode('utf-8')).hexdigest()
    if args.video_path.startswith('http://') or args.video_path.startswith('https://'):
        raise NotImplementedError("Supports only local files")
    else:
        video_file_path = args.video_path

    frames_cache_file = os.path.join(args.cache_dir, f'{video_hash}_{args.sample_freq}_frames.npy')
    timestamps_cache_file = os.path.join(args.cache_dir, f'{video_hash}_{args.sample_freq}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.arange(0,total_frames,args.sample_freq)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def numpy_to_bytes(image_np:np.ndarray):
    
    buffer = BytesIO()
    pil_image = Image.fromarray(image_np)
    pil_image.save(buffer, format="jpeg")
    
    return buffer.getvalue()

def frame_loader(args:PredictionConfig,img_as_bytes:bool=False):
    
    video_file_path, frames, timestamps = get_video_frames(args)
    
    func = lambda x:x
    if img_as_bytes:
        func = lambda x: numpy_to_bytes(x)
            
    num_frames = len(frames)
    batch = args.batch_frames
    for i in range(0,num_frames,batch):
        
        img_batch = [func(frames[i+k]) for k in range(min(num_frames,i+batch))]
                
        yield img_batch, timestamps[i:min(num_frames,i+batch)]


if __name__ == "__main__":
    
    args = PredictionConfig(video_path=r"D:\workspace\data\video\DJI_0023.MP4",
                            sample_freq=24,
                            cache_dir="../.cache",
                            batch_frames=5,
                            )
    
    loader = iter(frame_loader(args=args,img_as_bytes=True))
    
    # with open(args.video_path, 'rb') as f:
    #     vr = VideoReader(f, ctx=cpu(0))
    # frame = vr[0]
    
    frames, ts = next(loader)
    
    
    pass
    