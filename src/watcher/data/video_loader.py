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
        overlap_ratio: float = 0.2,
        batch_size: int = 1,
        max_frames: Optional[int] = None,
    ):
        
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.stride = int((1 - self.overlap_ratio) * self.tile_size)
        self.batch_size = batch_size
        self.max_frames = max_frames
            
    
    def load(self,video: Union[str, bytes, BytesIO],
            cache_dir: Optional[str] = None,
            sample_freq: int = 1,
            img_as_bytes: bool = True,
            batch_size: int = 1,
            max_frames: Optional[int] = None,
        ):
        """Load and process video frames"""
        frame_loader = self.frame_generator(
            video=video,
            cache_dir=cache_dir,
            sample_freq=sample_freq,
            img_as_bytes=img_as_bytes,
            batch_size=batch_size,
            max_frames=max_frames,
        )
        
        while True:
            data_package = self._load_once(frame_loader)
            if data_package == "DONE":
                break
            yield data_package
        
    
    def _get_patches(self, image: np.ndarray) -> np.ndarray:
        """Extract patches from image using numpy operations (no torch dependency)"""
        if image.ndim == 2:
            # Add channel dimension for grayscale
            image = image[..., np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False
        
        H, W, C = image.shape
        
        # Calculate number of patches
        h_patches = (H - self.tile_size) // self.stride + 1
        w_patches = (W - self.tile_size) // self.stride + 1
        
        patches = []
        
        for i in range(h_patches):
            for j in range(w_patches):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.tile_size
                w_end = w_start + self.tile_size
                
                patch = image[h_start:h_end, w_start:w_end, :]
                patches.append(patch)
        
        patches = np.array(patches)
        
        if squeeze_output:
            patches = patches.squeeze(axis=-1)
        
        return patches
    
    def _get_patches_from_frame(
        self, image: np.ndarray, patch_size: int
    ) -> tuple[np.ndarray, dict]:
        """Extract patches from the frame

        Args:
            image (np.ndarray): frame data
            patch_size (int): patch size

        Returns:
            tuple[np.ndarray, dict]: batch of patches, offset information
        """
        height, width = image.shape[:2]
        
        if width <= patch_size or height <= patch_size:
            LOGGER.debug("Image is too small for patch extraction")
            offset_info = {
                "y_offset": [0],
                "x_offset": [0],
                "y_end": [height],
                "x_end": [width],
            }
            return image[np.newaxis, ...], offset_info

        patches = self._get_patches(image)
        
        # Calculate offset information
        h_patches = (height - self.tile_size) // self.stride + 1
        w_patches = (width - self.tile_size) // self.stride + 1
        
        y_offsets = []
        x_offsets = []
        y_ends = []
        x_ends = []
        
        for i in range(h_patches):
            for j in range(w_patches):
                y_offset = i * self.stride
                x_offset = j * self.stride
                y_end = y_offset + patch_size
                x_end = x_offset + patch_size
                
                y_offsets.append(y_offset)
                x_offsets.append(x_offset)
                y_ends.append(y_end)
                x_ends.append(x_end)

        offset_info = {
            "y_offset": y_offsets,
            "x_offset": x_offsets,
            "y_end": y_ends,
            "x_end": x_ends,
        }

        return patches, offset_info
    
    def frame_generator(
        self,
        video: Union[str, bytes, BytesIO],
        cache_dir: Optional[str],
        sample_freq: int = 1,
        img_as_bytes: bool = True,
        batch_size: int = 1,
        max_frames: Optional[int] = None,
    ) -> Iterator[Tuple[list, np.ndarray]]:
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
            Tuple of (frame_batch, timestamps)
        """
        try:
            frames, timestamps = get_video_frames(
                video=video,
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
            
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                frame_batch = [convert_frame(frames[j]) for j in range(i, end_idx)]
                batch_timestamps = timestamps[i:end_idx]
                
                yield frame_batch, batch_timestamps
                
        except Exception as e:
            LOGGER.error(f"Error in frame_loader: {e}")
            raise
    
    def _load_once(self,frame_loader:Iterator[Tuple[list, np.ndarray]]) -> dict:
        """Load and process one frame"""
        try:
            frame, timestamp = next(frame_loader)

            # Extract patches from frame
            patches, offset_info = self._get_patches_from_frame(image=frame, 
            patch_size=self.tile_size)
                    
            data_package = {
                "timestamp":timestamp,
                "data": patches,
                "offset_info": offset_info,
            }
            return data_package

        except StopIteration:
            return "DONE"
        
        except Exception as e:
            LOGGER.error(f"Error in _load_once: {e}")
            raise
        

    
   
    