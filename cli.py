import fire
from watcher.analyzer.analysis import analyze_video
from watcher.config import PredictionConfig
from typing import Union, Optional
import json

def analyze(video: str, args: PredictionConfig, metadata: Optional[dict] = None) -> dict:
    args = PredictionConfig(**args)
    result = analyze_video(video=video, args=args, metadata=metadata)
    return json.dumps(vars(result))


if __name__ == "__main__":
    fire.Fire()