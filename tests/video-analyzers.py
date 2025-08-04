from dotenv import load_dotenv

from watcher.config import PredictionConfig
from watcher.data.video_loader import DataLoading
from watcher.analyzer import analyze_video

VIDEO_PATH = r"D:\workspace\data\video\DJI_0023.MP4"


if __name__ == "__main__":
    

    load_dotenv("../.env")

    args = PredictionConfig(clip_model="google/siglip2-base-patch16-224",
                            clip_device="auto",
                            clip_input_size=(1024,1024),
                            vlm_model=None,#"ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m",
                            llm_model="ggml-org/Qwen3-1.7B-GGUF:q4_k_m",
                            temperature=0.7,
                            img_as_bytes=True
                            )

    
    results = analyze_video(video=VIDEO_PATH,args=args,metadata=None,activity_analysis=True)

    print(results)
