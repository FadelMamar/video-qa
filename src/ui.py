import streamlit as st
import time
import io
from datetime import datetime
import random
import json
import os
from watcher.synthetizer import analyze_video
from watcher.config import PredictionConfig
from dotenv import load_dotenv

DOT_ENV = "../.env"



def main():
    # Configure page
    st.set_page_config(
        page_title="Video Analysis Demo",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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

    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")

        # Video upload
        uploaded_video = st.file_uploader(
            "Upload Video File",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supported formats: MP4, AVI, MOV, MKV, WEBM",
        )

        st.divider()

        # Analysis parameters
        with st.expander("⚙️ Advanced Settings"):
            sample_freq = st.number_input(
                "Sampling frequency",
                min_value=1,
                value=5,
                help="Downsampling rate. 5 means we retain 1 frame per 5 seconds.",
            )
            temperature = st.number_input(
                "Temperature",
                min_value=0.,
                value=0.7,
                help="Temperature of VLM",
            )
            batch_frames = st.number_input(
                "Batch frames",
                min_value=1,
                value=1,
                help="Number of frames to process in each batch. Higher values may speed up processing but require more memory.",
            )


        st.divider()

        # Analysis button
        analyze_button = st.button(
            "Start Analysis",
            type="primary",
            disabled=not uploaded_video,
            use_container_width=True,
        )

        analyze_button_2 = st.button(
            "Recognize activities",
            type="secondary",
            disabled=not uploaded_video,
            use_container_width=True,
        )

        if uploaded_video:
            st.success("✅ Ready to analyze!")
        else:
            if not uploaded_video:
                st.warning("⚠️ Please upload a video")

    (tab1,) = st.tabs(
            [
                "Video Analysis",
            ]
        )
    
    with tab1:
        # Main content area
        col1, col2 = st.columns([2, 1])

        with col1:
            # Video display section
            if uploaded_video:
                st.subheader("📹 Video Preview")
                st.video(uploaded_video)

                # Video info
                with st.expander("ℹ️ Video Information"):
                    st.write(f"**Filename:** {uploaded_video.name}")
                    st.write(f"**File size:** {uploaded_video.size / 1024 / 1024:.2f} MB")
                    st.write(f"**File type:** {uploaded_video.type}")
            else:
                st.info("👆 Please upload a video file using the sidebar to get started")

        with col2:
            # Quick info panel
            st.subheader("📊 Analysis Info")

            if uploaded_video:
                st.markdown(
                    f"""
                <div class="result-box">
                    <strong>Video Status:</strong> Ready<br>
                    <strong>Format:</strong> {uploaded_video.type}
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Analysis Results Section
        st.divider()

        if analyze_button and uploaded_video:
            st.subheader("🔍 Analysis Results")
            with st.spinner("Running..."):
                load_dotenv(DOT_ENV)
                args=PredictionConfig(video=uploaded_video,
                                    sample_freq=sample_freq,
                                    temperature=temperature,
                                    cache_dir="../.cache",
                                    model=os.getenv("MODEL_NAME","openai/Qwen2.5-VL-3B"),
                                    batch_frames=batch_frames,
                                    )
                

                frames_analysis, timestamps, analysis = analyze_video(args=args)

                # time.sleep(2)  # Simulate processing time
                # analysis = "There is a car driving on the road"
                # frames_analysis = ["Car detected", "No car", "Car detected"]
                # timestamps = [0, 5, 10]  # Simulated timestamps

                metadata = {f"Time: {timestamps[i]}s": frames_analysis[i] for i in range(len(timestamps))}

            # Detailed results tabs
            (tab1,) = st.tabs(
                [
                    "🎯 Key Insights",
                ]
            )

            with tab1:
                st.markdown("#### Main Findings")

                st.code(analysis, language="text")
                st.write(metadata)

                # Download button for results
                st.download_button(
                    label="📥 Download Analysis Results",
                    data=analysis,
                    file_name=f"analysis_{uploaded_video.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # Activity Recognition Section
    st.divider()
    
    if analyze_button_2 and uploaded_video:
        st.subheader("🏃 Activity Recognition")        
        with st.spinner("Analyzing activities..."):
            time.sleep(2)

        # Simulated activity recognition results
        activities = ["vehicles driving", "people walking", "animals running"]
        recog_activities = {f"Time: {random.randint(i,(i+10)**2)}s": activities[i] for i in range(len(activities))}

        st.write(recog_activities)

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


if __name__ == "__main__":
    main()
