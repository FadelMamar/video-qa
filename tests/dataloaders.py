from io import BytesIO


from watcher.agent.synthetizer import analyze_video
from watcher.config import PredictionConfig
from watcher.data.video_loader import DataLoading



if __name__ == "__main__":
    video_path = r"D:\workspace\data\video\DJI_0023.MP4"

    with open(video_path, 'rb') as f:
        video_buffer = BytesIO(f.read())
    
    handler = DataLoading()
    loader = handler.get_loader(video=video_buffer,
                                cache_dir=".cache",
                                sample_freq=5,
                                img_as_bytes=False,
                                max_frames=None)

    
    for data_package in loader:
        print(data_package["frame"].shape)
        break
