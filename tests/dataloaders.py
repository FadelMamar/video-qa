from io import BytesIO


from watcher.config import PredictionConfig
from watcher.utils import frame_loader



if __name__ == "__main__":
    video_path = r"D:\workspace\data\video\DJI_0023.MP4"
    with open(video_path, 'rb') as f:
        video_buffer = BytesIO(f.read())
    
    args = PredictionConfig(
        video_path=video_buffer,
        sample_freq=5,  # Sample every 'sample_freq' seconds
        cache_dir=None, #"../.cache",None        batch_frames=5,
    )

    loader = frame_loader(args=args, img_as_bytes=True)
    loader = iter(loader)

    # with open(args.video_path, 'rb') as f:
    #     vr = VideoReader(f, ctx=cpu(0))
    # frame = vr[0]

    frames, ts = next(loader)

    print(f"Loaded {len(frames)} frames with timestamps: {ts}")