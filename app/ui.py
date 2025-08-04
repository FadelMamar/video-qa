import streamlit as st
import time
import io
from datetime import datetime

import os
from typing import Union,List
from pathlib import Path
from watcher.config import PredictionConfig
from dotenv import load_dotenv
from watcher.base import FramesAnalysisResult,Frame
from watcher.detection.ultralytics_detector import create_yolo_detector

ROOT_DIR = Path(__file__).resolve().parent.parent
DOT_ENV = ROOT_DIR / ".env"

YOLO_MODEL_PATH = ROOT_DIR / "models" / "yoloe-11l-seg.pt"

def header():
    # Custom CSS for better styling
    st.markdown(
        """
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .analysis-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        .result-box {
            background-color: white;
            padding: 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .timestamp {
            color: #666;
            font-size: 0.9em;
            font-style: italic;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>Surveillance Video Analysis PoC</h1>
        <p>Leverage AI-powered insights to analyze surveillance videos</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    

def main():
    # Configure page
    st.set_page_config(
        page_title="Video Analysis Demo",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_dotenv(DOT_ENV)

    header()

    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")

        # Video upload
        uploaded_video = st.file_uploader(
            "Upload Video File",
            type=["mp4",],
            help="Supported formats: MP4"
        )
        with st.form("video_path_form"):
            video_path = st.text_input("Video path",value=r"D:\workspace\data\video\DJI_0023.MP4",placeholder="Enter video path for large videos 'mp4' only")
            button = st.form_submit_button("Load video",type="secondary",use_container_width=True)
            if button:
                video_path = Path(video_path.strip()).resolve()
                assert video_path.exists(), "Video path does not exist. Delete the quote marks and try again."

        st.divider()

        sample_freq = st.number_input(
            "Sampling frequency",
            min_value=1,
            value=5,
            help="Downsampling rate. 5 means we retain 1 frame per 5 seconds.",
            )   
        activity_analysis = st.checkbox("Fast analysis",value=True,help="Uses faster method for activity analysis")

        if uploaded_video:
            st.success("✅ Ready to analyze!")
        elif button:
            st.success("✅ Ready to analyze!")
        else:
            st.warning("⚠️ Please upload a video")
        
        if uploaded_video and video_path:
            st.warning("⚠️ Both video and video path are provided, discarding the uploaded video.")
            uploaded_video = None
        
        

    (tab1,tab0) = st.tabs(
            [
                "Video Analysis",
                "AI models",
                #"Reconnaissance"
            ]
        )

    with tab0:
        with st.form("vlm_form"):
            vlm_model = st.text_input("VLM Model path",
            #value="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m",
            placeholder="example: ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m",
            help="Enter the path to the VLM model."
            )
            llm_model = st.text_input("LLM model path",value="ggml-org/Qwen3-0.6B-GGUF:f16",placeholder="Enter LLM model path")
            clip_model = "google/siglip2-base-patch16-224" #st.text_input("CLIP model path",value="google/siglip2-base-patch16-224",placeholder="Enter CLIP model path")
            ctx_size = st.number_input("context size",value=20000,placeholder="Enter context size")
            
            vlm_button = st.form_submit_button("Launch AI models",use_container_width=True)

            if vlm_button:
                vlm_port = os.getenv("VLM_PORT")
                llm_port = os.getenv("LLM_PORT")
                if llm_model:
                    launch_llm_endpoint(model_name=llm_model,port=int(llm_port),ctx_size=ctx_size)
                    st.success("✅ Endpoint launched successfully")
    
                if vlm_model:
                    launch_llm_endpoint(model_name=vlm_model,port=int(vlm_port),ctx_size=ctx_size)
                    st.success("✅ Endpoint launched successfully")
                
    with tab1:
        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            # Video display section
            if uploaded_video:
                st.subheader("📹 Video")
                st.video(uploaded_video)  
            elif button:
                st.subheader("📹 Video")
                st.video(video_path,format="video/mp4")
            else:
                st.info("👆 Please upload a video file using the sidebar to get started")                            

        with col2:
            # Quick info panel
            st.subheader("ℹ️ Information")

            if uploaded_video:
                st.write(f"**Filename:** {uploaded_video.name}")
                st.write(f"**File size:** {uploaded_video.size / 1024 / 1024:.2f} MB")
                st.write(f"**File type:** {uploaded_video.type}")
            elif video_path:
                st.write(f"**Video path:** {video_path}")
            

        # Analysis Results Section
        st.divider()
        st.subheader("🎯 Video Analysis")
        with st.form("analysis_form"):
            analyze_button = st.form_submit_button(
            "Start Analysis",
            type="primary",
            disabled=(not uploaded_video) and (not video_path),
            use_container_width=True,
            )
            device = st.selectbox("device",options=["cpu","cuda"],index=0)
            
            temperature = float(os.getenv("TEMPERATURE",0.7))
            batch_frames = 1 
            #vlm_model=os.getenv("VLM_MODEL","ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:q4_k_m")
            #llm_model=os.getenv("LLM_MODEL","ggml-org/Qwen3-1.7B-GGUF:q4_k_m")

            if analyze_button and (uploaded_video or video_path):
                st.subheader("🔍 Analysis Results")
                with st.spinner("Running..."):
                    
                    args=PredictionConfig(sample_freq=sample_freq,
                                        temperature=temperature,
                                        vlm_model=vlm_model,
                                        llm_model=llm_model,
                                        batch_frames=batch_frames,
                                        clip_model=clip_model,
                                        clip_device=device,
                                        )
                    if uploaded_video:
                        video_content = uploaded_video.getvalue()
                    else:
                        with open(video_path,"rb") as f:
                            video_content = f.read()
                        result = analyze_video_cli(video=video_content, args=args,metadata=None,activity_analysis=activity_analysis)
                    
                    summary = result.get("summary") if isinstance(result, dict) else None
                    timestamps = result.get("timestamps") if isinstance(result, dict) else None
                    frames_analysis = result.get("frames_analysis") if isinstance(result, dict) else None
                    if timestamps and frames_analysis:
                        metadata = {f"Time: {timestamps[i]:.2f}s": frames_analysis[i] for i in range(len(timestamps))}
                    else:
                        metadata = None

                # Detailed results tabs
                (tab1,) = st.tabs(
                    [
                        "🎯 Key Insights",
                    ]
                )

                with tab1:
                    st.markdown("#### Main findings")
                    if summary:
                        st.write(summary)
                    if metadata:
                        with st.expander("Metadata"):
                            st.write(metadata)

    #with tab2:
        #st.subheader("🎯 Tracking")

        #with st.form("tracking_form"):
            #tracking_button = st.form_submit_button("Annotate frames",type="primary",use_container_width=True,disabled=not uploaded_video)
            #if tracking_button and uploaded_video:
                #with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    #tmpfile.write(uploaded_video.getvalue())
                    #video_path = tmpfile.name
                #with st.spinner("Annotating video..."):
                #    annotate_video(video_path,output_path="output.mp4",sliced=False,model_path=YOLO_MODEL_PATH)
                
                #st.video("output.mp4")
                #col1,col2 = st.columns(2)
                #with st.spinner("Annotating frames..."):
                #    frames = annotate_frames(video_path,sliced=True,model_path=YOLO_MODEL_PATH,sample_freq=sample_freq)
                #cols = st.columns(len(frames))
                #for i,frame in enumerate(frames):
                #    cols[i].image(frame.image)            

    # Footer
    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Ready for your next analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def analyze_video_cli(video: bytes, args: PredictionConfig, metadata=None,activity_analysis: bool = False) -> dict:
    import subprocess
    import json
    import tempfile
    from pathlib import Path
    import streamlit as st

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video)
        video_path = tmpfile.name

    cmd = [
        "python", "cli.py", "analyze",
        video_path,
        f"--args={vars(args)}", 
        f"--activity_analysis={activity_analysis}"
    ]
    if metadata is not None:
        cmd += [f"--metadata={json.dumps(metadata)}"]

    cwd = Path(__file__).parent.parent
    

    # Use Popen for live output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        cwd=cwd,
        bufsize=1,
        universal_newlines=True,
    )
    
    with st.expander("Logs"):
        log_placeholder = st.empty()
        logs = ""
        last_line = None
        for line in process.stdout:
            logs += line
            log_placeholder.code(logs)  # Update the Streamlit code block with new logs
            last_line = line

    process.stdout.close()
    process.wait()
    #print("return code",returncode)

    os.remove(video_path)

    try:
        return  json.loads(last_line)
    except Exception as e:
        print(e)

    return  {}

@st.cache_data(show_spinner="Annotating frames...")
def annotate_video(video_path: str, output_path: str,sliced:bool=False,model_path:str="yoloe-11s-seg.pt"):
    detector = create_yolo_detector(
        model_path=model_path,
        confidence_threshold=0.3,
        input_size=(640,640),
        device="cpu"  # or "cuda" if available
    )

    detector.inference_video(video_path,output_path=output_path,sliced=sliced)

@st.cache_data(show_spinner="Annotating frames...")
def annotate_frames(video_path:str,sliced:bool=True,model_path:str="yoloe-11s-seg.pt",sample_freq=5)->List[Frame]:
    from watcher.data import DataLoading
    from watcher.base import Video
    import uuid
    import torch

    data_loader = DataLoading()
    video = Video(video_path=video_path)
    loader = data_loader.get_loader(video,img_as_bytes=False,sample_freq=sample_freq)

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
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda" if torch.cuda.is_available() else "cpu"  # or "cuda" if available
    )

    frames = detector.annotate_frames(frames,sliced=sliced)
    return frames


def launch_llm_endpoint(model_name: str, port: int = 8000, ctx_size: int = 20000):
    import subprocess
    from pathlib import Path
    import os

    cwd = Path(__file__).resolve().parent.parent
    
    load_dotenv(DOT_ENV,override=True)

    if model_name:
        assert model_name.startswith("ggml-org/"), f"model_name must start with 'ggml-org/': {model_name}"
        os.environ["MODEL_NAME"] = model_name
    if port:
        assert isinstance(port, int), f"port must be an integer: {port} is {type(port)}"
        os.environ["PORT"] = str(port)
    if ctx_size:
        assert isinstance(ctx_size, int), f"ctx_size must be an integer: {ctx_size}"
        os.environ["CTX_SIZE"] = str(ctx_size)
    
    # Launch the batch file as a subprocess in the background
    llama_server_log = os.environ.get("LLAMA_SERVER_LOG","llama_server.log")
    with open(llama_server_log, "w") as f:
        subprocess.Popen(
            "launch_vlm_endpoint.bat",
            shell=False,
            cwd=cwd,
            #creationflags=subprocess.CREATE_NEW_CONSOLE,
            env=os.environ.copy(),
            stdout=f,
            stderr=f
        )
    
    with st.expander("Logs"):
        llama_server_log = str(os.environ.get("LLAMA_SERVER_LOG"))
        if os.path.exists(llama_server_log):
            with open(llama_server_log,"r") as f:
                st.code(f.read())
        else:
            st.warning("LLAMA_SERVER_LOG is not set in .env file, logs will not be displayed")

if __name__ == "__main__":
    main()
