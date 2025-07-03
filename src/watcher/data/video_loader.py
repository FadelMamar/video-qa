import os
import hashlib
from PIL import Image
from io import BytesIO
import numpy as np
import traceback
from typing import Union, Tuple, Iterator, Optional, List
from pathlib import Path
import logging
from decord import VideoReader, cpu
from tqdm import tqdm

from ..base import Video
from ..config import PredictionConfig


# Set up logging
LOGGER = logging.getLogger(__name__)


def calculate_video_hash(video: Union[str, bytes, BytesIO]) -> str:
    """Calculate hash for video input with improved efficiency."""
    if isinstance(video, str):
        if video.startswith(("http://", "https://")):
            raise NotImplementedError("Supports only local files")
        return hashlib.md5(video.encode("utf-8")).hexdigest()
    
    elif isinstance(video, bytes):
        return hashlib.md5(video).hexdigest()
    
    elif isinstance(video, BytesIO):
        # More efficient: read in chunks instead of loading entire content
        hash_md5 = hashlib.md5()
        video.seek(0)  # Reset position
        for chunk in iter(lambda: video.read(4096), b""):
            hash_md5.update(chunk)
        video.seek(0)  # Reset position for later use
        return hash_md5.hexdigest()
    
    else:
        raise ValueError(
            "video must be a string (file path), bytes (file content) or io.BytesIO (in-memory file)"
        )


def validate_video_file(video: Union[str, bytes, BytesIO]) -> bool:
    """Validate that the video file can be opened and read."""
    try:
        if isinstance(video, str):
            if not os.path.exists(video):
                return False
            # Try to open with VideoReader to validate format
            vr = VideoReader(video, ctx=cpu(0))
            return len(vr) > 0
        else:
            # For bytes or BytesIO, try to create VideoReader
            vr = VideoReader(video, ctx=cpu(0))
            return len(vr) > 0
    except Exception as e:
        LOGGER.warning(f"Video validation failed: {e}")
        return False


def get_video_frames(
    video: Union[str, bytes, BytesIO],
    cache_dir: Optional[str],
    sample_freq: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract frames from video with improved memory management and caching.
    
    Args:
        video: Video source (file path, bytes, or BytesIO)
        cache_dir: Directory for caching frames
        sample_freq: Sample every N seconds
        max_frames: Maximum number of frames to extract (None for all)
    
    Returns:
        Tuple of (frames, timestamps)
    """
    # Validate input
    if not validate_video_file(video):
        raise ValueError("Invalid or corrupted video file")
    
    # Calculate hash
    video_hash = calculate_video_hash(video)
    
    # Setup cache files
    frames_cache_file = ""
    timestamps_cache_file = ""
    
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        frames_cache_file = cache_path / f"{video_hash}_{sample_freq}_frames.npy"
        timestamps_cache_file = cache_path / f"{video_hash}_{sample_freq}_timestamps.npy"
    
    # Return cached frames if they exist
    if frames_cache_file and timestamps_cache_file:
        if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
            LOGGER.info("Loading frames from cache")
            frames = np.load(frames_cache_file)
            timestamps = np.load(timestamps_cache_file)
            return frames, timestamps
    
    # Load video
    LOGGER.info("Loading video frames")
    vr = VideoReader(video, ctx=cpu(0))
    total_frames = len(vr)
    
    if total_frames == 0:
        raise ValueError("Video contains no frames")
    
    # Calculate sampling indices
    fps = vr.get_avg_fps()
    if fps <= 0:
        raise ValueError("Invalid video FPS")
    
    step = int(fps * sample_freq)
    if step <= 0:
        step = 1
    
    indices = np.arange(0, total_frames, step)
    
    # Apply max_frames limit
    LOGGER.info(f"max_frames: {max_frames}, total_frames: {total_frames}")
    if max_frames:
        max_frames = min(max_frames, total_frames)
        indices = indices[:max_frames]
    
    LOGGER.info(f"Extracting {len(indices)} frames from {total_frames} total frames")
    
    # Extract frames in batches for better memory management
    batch_size = min(100, len(indices))  # Process in smaller batches
    frames_list = []
    timestamps_list = []
    
    for i in tqdm(range(0, len(indices), batch_size), desc="Extracting frames"):
        batch_indices = indices[i:i + batch_size]
        
        # Extract frames
        batch_frames = vr.get_batch(batch_indices).asnumpy()
        frames_list.append(batch_frames)
        
        # Calculate timestamps more efficiently
        batch_timestamps = np.array([
            vr.get_frame_timestamp(idx) for idx in batch_indices
        ])
        
        timestamps_list.append(batch_timestamps)
    
    # Combine batches
    frames = np.concatenate(frames_list, axis=0)
    timestamps = np.concatenate(timestamps_list, axis=0)
    
    # Cache results
    if cache_dir:
        LOGGER.info("Caching frames and timestamps")
        np.save(frames_cache_file, frames)
        np.save(timestamps_cache_file, timestamps)
    
    LOGGER.info(f"Successfully extracted {len(frames)} frames")
    return frames, timestamps


def numpy_to_bytes(image_np: np.ndarray, save_as: str = "jpeg", quality: int = 85) -> bytes:
    """
    Convert numpy array to bytes with configurable quality.
    
    Args:
        image_np: Input numpy array
        save_as: Image format (jpeg, png, etc.)
        quality: JPEG quality (1-100, only for JPEG)
    
    Returns:
        Image as bytes
    """
    if not isinstance(image_np, np.ndarray):
        raise ValueError(f"Expected type 'np.ndarray' but received {type(image_np)}")
    
    buffer = BytesIO()
    pil_image = Image.fromarray(image_np)
    
    # Configure save parameters
    if save_as.lower() == "jpeg":
        pil_image.save(buffer, format=save_as, quality=quality, optimize=True)
    else:
        pil_image.save(buffer, format=save_as)
    
    return buffer.getvalue()


def get_video_info(video: Union[str, bytes, BytesIO]) -> dict:
    """
    Get video metadata without loading all frames.
    
    Args:
        video: Video source
    
    Returns:
        Dictionary with video information
    """
    try:
        vr = VideoReader(video, ctx=cpu(0))
        return {
            "total_frames": len(vr),
            "fps": vr.get_avg_fps(),
            "duration": len(vr) / vr.get_avg_fps() if vr.get_avg_fps() > 0 else 0,
            "width": vr[0].shape[1] if len(vr) > 0 else 0,
            "height": vr[0].shape[0] if len(vr) > 0 else 0,
        }
    except Exception as e:
        LOGGER.error(f"Error getting video info: {e}")
        raise


class DataLoading:
    """
    Class responsible for loading and preparing video data with tiling capabilities
    """

    def __init__(
        self,
        tile_size: int = 800,
        load_patches: bool = False,
        overlap_ratio: float = 0.2,
        batch_size: int = 1,
        max_frames: Optional[int] = None,
    ):
        
        self.tile_size = tile_size
        self.load_patches = load_patches
        self.overlap_ratio = overlap_ratio
        self.stride = int((1 - self.overlap_ratio) * self.tile_size)
        self.batch_size = batch_size
        self.max_frames = max_frames
            
    
    def get_loader(self,video: Video,
            cache_dir: Optional[str] = None,
            sample_freq: int = 1,
            img_as_bytes: bool = True,
            max_frames: Optional[int] = None,
        ):
        """Load and process video frames"""
        
        return self.frame_generator(
            video=video,
            cache_dir=cache_dir,
            sample_freq=sample_freq,
            img_as_bytes=img_as_bytes,
            max_frames=max_frames,
        )
            
    def frame_generator(
        self,
        video: Video,
        cache_dir: Optional[str],
        sample_freq: int = 1,
        img_as_bytes: bool = True,
        max_frames: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Generator for loading video frames in batches with improved memory management.
        
        Args:
            video: Video source
            cache_dir: Directory for caching frames
            sample_freq: Sample every N seconds
            img_as_bytes: Whether to convert frames to bytes
            batch_size: Number of frames per batch
            max_frames: Maximum number of frames to load
        
        Yields:
            Dictionary with timestamp, data, and offset_info
        """

        frames, timestamps = get_video_frames(
            video=video.video_path,
            cache_dir=cache_dir,
            sample_freq=sample_freq,
            max_frames=max_frames,
        )
        
        # Setup conversion function
        if img_as_bytes:
            def convert_to_bytes(frame: np.ndarray) -> bytes:
                return numpy_to_bytes(frame, save_as="jpeg")
            convert_frame = convert_to_bytes
        else:
            def convert_to_array(frame: np.ndarray) -> np.ndarray:
                return frame
            convert_frame = convert_to_array

        # Yield frames in batches
        num_frames = len(frames)
        
        for i in range(0, num_frames):
            frame = convert_frame(frames[i])
            timestamp = timestamps[i].tolist()

            data_package = {
                "timestamp":timestamp,
                "frame": frame,
                "offset_info": None,
            }
            
            yield data_package
                
        
            

    
   
    