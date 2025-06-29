import io


from .vlm import DspyAnalyzer, Summarizer
from ..config import PredictionConfig
from ..utils import frame_loader

def analyze_video(
    args:PredictionConfig,
) -> list[str]:
    assert isinstance(args.video, io.BytesIO), (
        f"Expected type 'io.BytesIO' but received {type(args.video)} "
    )

    handler_analyze = DspyAnalyzer(model=args.model,
                                   temperature=args.temperature,
                                   prompting_mode=args.prompting_mode,
                                   cache=True
                                   )
    handler_summary = Summarizer(
        model=args.model, 
        temperature=args.temperature,
        cache=True
    )

    # Analyze frames
    loader = frame_loader(args=args, img_as_bytes=True)
    frames_analysis = []
    timestamps = []

    for frames, ts in loader:
        o = handler_analyze.run(frames)
        frames_analysis.append(o)
        timestamps.append(ts)

    # Summarize analysis
    response = handler_summary.run(frames_analysis)

    return frames_analysis, timestamps, response