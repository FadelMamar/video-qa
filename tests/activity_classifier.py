from watcher.analyzer.clip_analyzer import create_clip_classifier
from watcher.data.video_loader import DataLoading
from watcher.base import Video

VIDEO_PATH = r"D:\workspace\data\video\DJI_0023.MP4"
MODEL_PATH="google/siglip2-base-patch16-224"
MODEL_PATH = r"D:\workspace\repos\video-qa\models\models--chendelong--RemoteCLIP\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38\RemoteCLIP-ViT-B-32.pt"

def run_clip_classifier():

    clip_classifier = create_clip_classifier(
        model_path=MODEL_PATH,
        device="auto",
        input_size=(1024,1024),
        remote_clip=True
    )

    data_loader = DataLoading()
    video = Video(video_path=VIDEO_PATH)
    loader = data_loader.get_loader(video,img_as_bytes=True,sample_freq=5)

    images = [next(loader)["frame"] for _ in range(1)]

    result = clip_classifier(images)

    #print(result[0].activity_type,result[0].confidence)
    print(result)
   



if __name__ == "__main__":
    run_clip_classifier()
    

