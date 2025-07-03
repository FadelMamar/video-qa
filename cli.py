import fire
from watcher.analyzer.analysis import analyze_video
from watcher.config import PredictionConfig
from typing import Union, Optional
import json

def analyze(video: str, args: dict, metadata: Optional[dict] = None,activity_analysis: bool = False) -> dict:
    args = PredictionConfig(**args)
    result = analyze_video(video=video, args=args, metadata=metadata,activity_analysis=activity_analysis)
    return json.dumps(vars(result))


if __name__ == "__main__":
    fire.Fire()