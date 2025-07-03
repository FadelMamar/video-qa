import numpy as np
from watcher.detection.ultralytics_detector import create_yolo_detector
from watcher.base import Frame,Video
from watcher.data.video_loader import DataLoading
from watcher.base import ACTIVITY_PROMPTS
import uuid

# === User: Set your YOLO model path here ===
MODEL_PATH = "yoloe-11s-seg.pt"  # <-- Change this to your model file
VIDEO_PATH = r"D:\workspace\data\video\DJI_0023.MP4"

def run():
    
    data_loader = DataLoading()
    video = Video(video_path=VIDEO_PATH)
    loader = data_loader.get_loader(video,img_as_bytes=False,sample_freq=5)

    # Create a Frame object (detections is empty before inference)
    frames = []
    for data in loader:
        dummy_frame = Frame(
            timestamp=data["timestamp"],
            image=data["frame"],
            detections=[],
            parent_video_id=str(uuid.uuid4()),
        )
        frames.append(dummy_frame)

    # Create the detector (update model_path as needed)
    detector = create_yolo_detector(
        model_path=MODEL_PATH,
        confidence_threshold=0.3,
        device="cpu"  # or "cuda" if available
    )

    # Run inference on the dummy frame
    try:
        results = detector.inference(frames,sliced=True)
        print(results)
        #print("Detections for dummy frame:")
        for result in results:
            print(result.detections,result.timestamp)
        #for det in results[0].detections:
        #    print(f"Class: {det.class_name}, Confidence: {det.confidence:.2f}, BBox: {det.bbox}")
    except Exception as e:
        print(f"Error during inference: {e}")

    # === User: To test on real data, replace dummy_image and MODEL_PATH accordingly ===

def run_video():
    detector = create_yolo_detector(
        model_path=MODEL_PATH,
        confidence_threshold=0.3,
        input_size=(640,640),
        device="cpu"  # or "cuda" if available
    )

    detector.inference_video(VIDEO_PATH,output_path="output.mp4",sliced=False)

if __name__ == "__main__":
    run_video()
    #run()