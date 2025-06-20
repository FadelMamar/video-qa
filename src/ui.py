import streamlit as st
import time
import io
from datetime import datetime


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
        
        # Analysis instruction
        analysis_instruction = st.text_area(
            "Analysis Instruction",
            placeholder="e.g., 'Analyze the speaker's main arguments and identify key topics discussed'",
            height=100,
            help="Describe what kind of analysis you want to perform on the video"
        )
        
        # Analysis parameters
        with st.expander("⚙️ Advanced Settings"):
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
            include_timestamps = st.checkbox("Include Timestamps", value=True)
            detailed_analysis = st.checkbox("Detailed Analysis", value=False)
        
        st.divider()
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Start Analysis",
            type="primary",
            disabled=not (uploaded_video and analysis_instruction),
            use_container_width=True
        )
        
        if uploaded_video and analysis_instruction:
            st.success("✅ Ready to analyze!")
        else:
            if not uploaded_video:
                st.warning("⚠️ Please upload a video")
            if not analysis_instruction:
                st.warning("⚠️ Please provide analysis instruction")
    
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
        if analysis_instruction:
            st.markdown(f"""
            <div class="result-box">
                <strong>Current Instruction:</strong><br>
                {analysis_instruction}
            </div>
            """, unsafe_allow_html=True)
        
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
    
    if analyze_button and uploaded_video and analysis_instruction:
        st.subheader("🔍 Analysis Results")
        
        # Progress bar simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate analysis process
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🎥 Processing video frames...")
            elif i < 60:
                status_text.text("🎯 Extracting features...")
            elif i < 90:
                status_text.text("🧠 Analyzing content...")
            else:
                status_text.text("📝 Generating results...")
            time.sleep(0.02)
        
        status_text.text("✅ Analysis complete!")
        progress_bar.empty()
        
        # Mock analysis results
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        
        # Summary section
        st.markdown("### 📋 Executive Summary")
        st.markdown("""
        <div class="result-box">
            <strong>Analysis completed successfully!</strong><br><br>
            Based on your instruction: "<em>{}</em>"<br><br>
            Key findings and insights have been generated below. The analysis identified several important 
            segments and topics throughout the video content.
        </div>
        """.format(analysis_instruction), unsafe_allow_html=True)
        
        # Detailed results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Key Insights", "📍 Segments", "📊 Statistics", "📄 Raw Data"])
        
        with tab1:
            st.markdown("#### Main Findings")
            findings = [
                "Primary topic identified with 94% confidence",
                "3 distinct segments detected in the video",
                "Key themes: technology, innovation, future trends",
                "Speaker engagement level: High throughout presentation"
            ]
            
            for i, finding in enumerate(findings, 1):
                st.markdown(f"""
                <div class="result-box">
                    <strong>Finding {i}:</strong> {finding}
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("#### Timeline Analysis")
            
            if include_timestamps:
                segments = [
                    {"time": "00:00 - 02:30", "topic": "Introduction and Context", "confidence": 0.92},
                    {"time": "02:30 - 07:15", "topic": "Main Discussion Points", "confidence": 0.87},
                    {"time": "07:15 - 09:45", "topic": "Examples and Case Studies", "confidence": 0.91},
                    {"time": "09:45 - 12:00", "topic": "Conclusion and Summary", "confidence": 0.89}
                ]
                
                for segment in segments:
                    confidence_color = "green" if segment["confidence"] > 0.8 else "orange"
                    st.markdown(f"""
                    <div class="result-box">
                        <strong>{segment['time']}</strong><br>
                        Topic: {segment['topic']}<br>
                        <span style="color: {confidence_color}">Confidence: {segment['confidence']:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("#### Analysis Statistics")
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("Total Segments", "4", "100%")
                st.metric("Average Confidence", "89.7%", "2.3%")
            
            with col_stats2:
                st.metric("Processing Time", "2.1s", "-0.3s")
                st.metric("Key Topics Found", "8", "3")
            
            # Simple chart simulation
            chart_data = {
                'Segment': ['Intro', 'Main', 'Examples', 'Conclusion'],
                'Confidence': [92, 87, 91, 89]
            }
            st.bar_chart(chart_data, x='Segment', y='Confidence')
        
        with tab4:
            st.markdown("#### Raw Analysis Output")
            
            raw_output = f"""
    Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Video File: {uploaded_video.name}
    Instruction: {analysis_instruction}
    Confidence Threshold: {confidence_threshold}
            
    ========== DETAILED ANALYSIS ==========
            
    Segment 1 (00:00-02:30):
    - Topic: Introduction and Context
    - Confidence: 92%
    - Key phrases: ["welcome", "today we'll discuss", "important topic"]
            
    Segment 2 (02:30-07:15):
    - Topic: Main Discussion Points  
    - Confidence: 87%
    - Key phrases: ["first point", "consider this", "important aspect"]
            
    Segment 3 (07:15-09:45):
    - Topic: Examples and Case Studies
    - Confidence: 91%
    - Key phrases: ["for example", "case study", "real-world application"]
            
    Segment 4 (09:45-12:00):
    - Topic: Conclusion and Summary
    - Confidence: 89%
    - Key phrases: ["in conclusion", "to summarize", "key takeaways"]
            
    ========== END ANALYSIS ==========
            """
            
            st.code(raw_output, language="text")
            
            # Download button for results
            st.download_button(
                label="📥 Download Analysis Results",
                data=raw_output,
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