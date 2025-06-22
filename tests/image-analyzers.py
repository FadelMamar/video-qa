from dotenv import load_dotenv

from watcher.config import PredictionConfig
from watcher.utils import frame_loader
from watcher.vlm import DspyAnalyzer, Summarizer


if __name__ == "__main__":
    

    load_dotenv("../.env")

    lm = DspyAnalyzer(model="openai/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                      prompting_mode="basic",
                     temperature=0.7, 
                     cache=False)

    args = PredictionConfig(video_path=r"D:\workspace\data\video\DJI_0023.MP4",
                            sample_freq=5,
                            cache_dir="../.cache",
                            batch_frames=2,
                            )

    loader = iter(frame_loader(args=args,img_as_bytes=True))

    # for _ in range(3):
    #     next(loader)

    frames, ts = next(loader)

    output = lm.run(frames)

    print(output)

# output = lm.inference_with_api(images=frames,as_video=False)