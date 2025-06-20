import streamlit as st
import time
import io
from datetime import datetime

import os
from dotenv import load_dotenv

from utils import frame_loader
from config import PredictionConfig
from vlm import DspyAnalyzer, Summarizer

DOT_ENV = "../.env"


def analyze_video(video_buffer:io.BytesIO,
                  sample_freq:int=24,
                  batch_frames:int=2)->list[str]:
    
    assert isinstance(video_buffer, io.BytesIO), "Expected type 'io.BytesIO' but received {type(video_buffer)} "
    
    load_dotenv(DOT_ENV)
    
    handler_analyze = DspyAnalyzer(model="openai/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF")
    handler_summary = Summarizer(model="openai/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                                 temperature=0.1)
    
    args = PredictionConfig(video_path=video_buffer.get_value(),
                            sample_freq=sample_freq,
                            cache_dir="../.cache",
                            batch_frames=batch_frames,
                            )
    
    # Analyze frames
    loader = frame_loader(args=args,img_as_bytes=True)
    out = []
    for frames, ts in loader:
        o = handler_analyze.run(frames)
        out.append(o)
    
    # Summarize analysis
    response = handler_summary(out)
    
    return response

def main():
    # Configure page
    st.set_page_config(
        page_title="Video Analysis Demo",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
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
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Video Analysis Demo</h1>
        <p>Upload a video and provide analysis instructions to see AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Video upload
        uploaded_video = st.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
        )
        
        st.divider()
        
                
        # Analysis parameters
        with st.expander("⚙️ Advanced Settings"):
            sample_freq = st.number_input("Sampling frequency",
                                          min_value=1,
                                          value=24,
                                          help="Downsampling rate. 24 means we retain 1 frame for 24 frames.")
                    
        st.divider()
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Start Analysis",
            type="primary",
            disabled=not uploaded_video,
            use_container_width=True
        )
        
                
        if uploaded_video:
            st.success("✅ Ready to analyze!")
        else:
            if not uploaded_video:
                st.warning("⚠️ Please upload a video")
                
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
            st.markdown(f"""
            <div class="result-box">
                <strong>Video Status:</strong> Ready<br>
                <strong>Duration:</strong> Processing...<br>
                <strong>Format:</strong> {uploaded_video.type}
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis Results Section
    st.divider()    
    
    
    if analyze_button and uploaded_video:
        st.subheader("🔍 Analysis Results")
        with st.spinner("Running..."):
            
            analysis = analyze_video(uploaded_video,sample_freq=sample_freq)        
        
        
        # Detailed results tabs
        tab1, = st.tabs(["🎯 Key Insights",])
        
        with tab1:
            st.markdown("#### Main Findings")
                        
                        
            st.code(analysis, language="text")
            
            # Download button for results
            st.download_button(
                label="📥 Download Analysis Results",
                data=analysis,
                file_name=f"analysis_{uploaded_video.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Video Analysis Demo • Built with Streamlit • 
        <span class="timestamp">Ready for your next analysis</span></p>
    </div>
    """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    
    main()