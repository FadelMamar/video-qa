import streamlit as st
import time
import io
from datetime import datetime
import random
import json
import os
from typing import Union
#from watcher.analyzer import analyze_video
from watcher.config import PredictionConfig
from dotenv import load_dotenv
from watcher.base import FramesAnalysisResult

DOT_ENV = "../.env"

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
        <h1>🎬 Video Analysis Demo</h1>
        <p>Upload a video and provide analysis instructions to see AI-powered insights</p>
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
        else:
            st.warning("⚠️ Please upload a video")

    (tab1,tab2) = st.tabs(
            [
                "Video Analysis",
                "Reconnaissance"
            ]
        )
    
    with tab1:
        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            # Video display section
            if uploaded_video:
                st.subheader("📹 Video")
                st.video(uploaded_video)  
            else:
                st.info("👆 Please upload a video file using the sidebar to get started")                            

        with col2:
            # Quick info panel
            st.subheader("ℹ️ Information")

            if uploaded_video:
                st.write(f"**Filename:** {uploaded_video.name}")
                st.write(f"**File size:** {uploaded_video.size / 1024 / 1024:.2f} MB")
                st.write(f"**File type:** {uploaded_video.type}")
            

        # Analysis Results Section
        st.divider()
        st.subheader("🎯 Video Analysis")
        with st.form("analysis_form"):
            analyze_button = st.form_submit_button(
            "Start Analysis",
            type="primary",
            disabled=not uploaded_video,
            use_container_width=True,
            )
            
            temperature = float(os.getenv("TEMPERATURE",0.7))
            batch_frames = 1 
            model=os.getenv("MODEL_NAME","openai/Qwen2.5-VL-3B")


            if analyze_button and uploaded_video:
                st.subheader("🔍 Analysis Results")
                with st.spinner("Running..."):
                    
                    args=PredictionConfig(sample_freq=sample_freq,
                                        temperature=temperature,
                                        vlm_model=model,
                                        batch_frames=batch_frames,
                                        )
                    
                    result = analyze_video_cli(video=uploaded_video.getvalue(), args=args,metadata=None,activity_analysis=activity_analysis)
                    
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


def analyze_video_cli(video: bytes, args: PredictionConfig, metadata=None,activity_analysis: bool = False) -> FramesAnalysisResult:
    import subprocess
    import json
    import tempfile
    from pathlib import Path
    import streamlit as st

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video)
        video_path = tmpfile.name

    cmd = [
        "uv", "run", "cli.py", "analyze",
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
    returncode = process.wait()
    print("return code",returncode)
    try:
        return  json.loads(last_line)
    except Exception as e:
        print(e)

    return  {}
    
    
if __name__ == "__main__":
    main()
