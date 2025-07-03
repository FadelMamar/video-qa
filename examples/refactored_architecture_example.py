"""
Example demonstrating the refactored architecture with clean separation
between analyzer and activity modules.

This example shows how the analyzer module provides the interface and orchestration,
while the activity module focuses on specific activity recognition strategies.
"""

import io
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_analyzer_orchestration():
    """Example of using analyzer module for general video analysis."""
    print("=== Analyzer Module: General Video Analysis ===")
    
    from src.watcher.analyzer.analysis import VideoSynthesizer
    from src.watcher.config import PredictionConfig
    
    # Create configuration for general video analysis
    config = PredictionConfig(
        model="openai/gpt-4-vision-preview",
        temperature=0.7,
        sample_freq=1,
        max_frames=10,
    )
    
    # Add analyzer type
    setattr(config, 'analyzer_type', 'dspy')
    
    # Create VideoSynthesizer for general analysis
    synthesizer = VideoSynthesizer(config)
    
    print("VideoSynthesizer created for general video analysis")
    print("This handles: frame extraction, analysis, summarization")
    
    # Example usage (commented out since we don't have actual video data)
    # with open("path/to/video.mp4", "rb") as f:
    #     video_data = io.BytesIO(f.read())
    #     result = synthesizer.analyze_video(video_data)
    #     print(f"General analysis summary: {result.summary}")


def example_activity_recognition_strategies():
    """Example of using activity module for specific activity recognition."""
    print("\n=== Activity Module: Activity Recognition Strategies ===")
    
    from src.watcher.activity.analyzer_strategy import (
        AnalyzerStrategyConfig, 
        create_analyzer_strategy
    )
    from src.watcher.activity.base import TrackedObject, ActivityType
    from src.watcher.activity.recognizer import DroneActivityRecognizer
    
    # Strategy 1: HF-based activity recognition
    print("\n--- Strategy 1: HF-based Activity Recognition ---")
    hf_config = AnalyzerStrategyConfig(
        analyzer_type="hf",
        model_name="Salesforce/blip-image-captioning-base",
        device="auto",
        max_new_tokens=30,
        confidence_threshold=0.6,
        ambiguous_threshold=0.3
    )
    
    hf_strategy = create_analyzer_strategy(hf_config)
    print(f"Created HF strategy: {type(hf_strategy).__name__}")
    
    # Strategy 2: DSPy-based activity recognition
    print("\n--- Strategy 2: DSPy-based Activity Recognition ---")
    dspy_config = AnalyzerStrategyConfig(
        analyzer_type="dspy",
        model_name="openai/gpt-4-vision-preview",
        confidence_threshold=0.7,
        ambiguous_threshold=0.4
    )
    
    dspy_strategy = create_analyzer_strategy(dspy_config)
    print(f"Created DSPy strategy: {type(dspy_strategy).__name__}")
    
    # Strategy 3: Traditional VLM classifier (legacy)
    print("\n--- Strategy 3: Traditional VLM Classifier (Legacy) ---")
    from src.watcher.activity.vlm_classifier import VLMClassifier
    
    # This would use the original VLM classifier
    # vlm_classifier = VLMClassifier(model_path="path/to/model")
    print("Traditional VLM classifier available for backward compatibility")
    
    # Example tracked object (simulated)
    tracked_obj = TrackedObject(
        track_id=1,
        class_name="person",
        bboxes=[(100, 100, 200, 300), (110, 105, 210, 305)],
        confidences=[0.9, 0.85],
        timestamps=[1.0, 2.0],
        frame_ids=[1, 2]
    )
    
    print(f"Sample tracked object: {tracked_obj.class_name} (ID: {tracked_obj.track_id})")


def example_integrated_workflow():
    """Example of integrated workflow using both modules."""
    print("\n=== Integrated Workflow: Analyzer + Activity ===")
    
    from src.watcher.activity.analyzer_strategy import AnalyzerStrategyConfig
    from src.watcher.activity.recognizer import DroneActivityRecognizer
    
    # Step 1: Configure activity recognition with analyzer backend
    activity_config = AnalyzerStrategyConfig(
        analyzer_type="hf",
        model_name="Salesforce/blip-image-captioning-base",
        confidence_threshold=0.6
    )
    
    # Step 2: Create activity recognizer with analyzer strategy
    activity_recognizer = DroneActivityRecognizer(
        analyzer_strategy_config=activity_config,
        sequence_length=8,
        input_size=(224, 224)
    )
    
    print("Created integrated activity recognizer")
    print("This uses analyzer module for image analysis")
    print("And activity module for activity classification")
    
    # Example workflow (commented out since we don't have actual data)
    # frames = load_video_frames("path/to/video.mp4")
    # tracked_objects = detect_and_track_objects(frames)
    # 
    # # Extract object clips
    # object_clips = activity_recognizer.extract_object_clips(frames, tracked_objects)
    # 
    # # Classify activities
    # individual_results, group_results = activity_recognizer.process_tracked_objects(
    #     frames, tracked_objects
    # )
    # 
    # print(f"Detected {len(individual_results)} individual activities")
    # print(f"Detected {len(group_results)} group activities")


def example_direct_analyzer_usage():
    """Example of using analyzer directly for activity recognition."""
    print("\n=== Direct Analyzer Usage for Activity Recognition ===")
    
    from src.watcher.analyzer.hf_analyzer import create_hf_analyzer
    from src.watcher.analyzer.dspy_analyzer import DSPyFrameAnalyzer
    from src.watcher.analyzer.base import ModelConfig
    from src.watcher.activity.base import ActivityType, TrackedObject
    
    # Create HF analyzer directly
    hf_analyzer = create_hf_analyzer(
        model_name="Salesforce/blip-image-captioning-base",
        device="auto",
        max_new_tokens=30
    )
    
    # Create DSPy analyzer directly
    dspy_config = ModelConfig(
        model_name="openai/gpt-4-vision-preview",
        temperature=0.7
    )
    dspy_analyzer = DSPyFrameAnalyzer(
        model_config=dspy_config,
        prompting_mode="basic"
    )
    
    print("Created analyzers directly:")
    print(f"- HF Analyzer: {type(hf_analyzer).__name__}")
    print(f"- DSPy Analyzer: {type(dspy_analyzer).__name__}")
    
    # Example usage (commented out since we don't have actual data)
    # image_bytes = load_image_as_bytes("path/to/image.jpg")
    # 
    # # General analysis
    # general_result = hf_analyzer.analyze(image_bytes)
    # print(f"General analysis: {general_result}")
    # 
    # # Activity-specific analysis
    # activity_result = hf_analyzer.analyze_activity(
    #     image_bytes, 
    #     activity_type=ActivityType.NORMAL_WALKING
    # )
    # print(f"Activity analysis: {activity_result}")
    # 
    # # Tracked object analysis
    # tracked_obj = TrackedObject(...)
    # object_result = hf_analyzer.analyze_tracked_object([image_bytes], tracked_obj)
    # print(f"Object analysis: {object_result}")


def example_architecture_comparison():
    """Show the architectural differences between old and new approaches."""
    print("\n=== Architecture Comparison ===")
    
    print("\nOLD ARCHITECTURE (Overlapping):")
    print("- Analyzer: Basic image analysis")
    print("- Activity: Duplicate VLM functionality")
    print("- Overlap: Both modules had similar capabilities")
    print("- Issues: Code duplication, unclear responsibilities")
    
    print("\nNEW ARCHITECTURE (Clean Separation):")
    print("- Analyzer: Generic image/video analysis interface and orchestration")
    print("  * BaseAnalyzer: Common interface for all analyzers")
    print("  * VideoSynthesizer: Orchestrates video analysis pipeline")
    print("  * HF/DSPy Analyzers: Specific implementation backends")
    print("  * Activity support: Built-in methods for activity recognition")
    
    print("\n- Activity: Specialized activity recognition strategies")
    print("  * ActivityClassifier: Base interface for activity recognition")
    print("  * AnalyzerActivityStrategy: Uses analyzer module for analysis")
    print("  * VLMClassifier: Legacy implementation for backward compatibility")
    print("  * DroneActivityRecognizer: Orchestrates activity recognition pipeline")
    
    print("\nBENEFITS:")
    print("- Clear separation of concerns")
    print("- Reusable analyzer components")
    print("- Flexible activity recognition strategies")
    print("- Backward compatibility")
    print("- Easier testing and maintenance")


if __name__ == "__main__":
    print("Refactored Architecture Examples")
    print("=" * 50)
    
    try:
        example_analyzer_orchestration()
        example_activity_recognition_strategies()
        example_integrated_workflow()
        example_direct_analyzer_usage()
        example_architecture_comparison()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey Points:")
        print("1. Analyzer module provides the interface and orchestration")
        print("2. Activity module focuses on specific activity recognition strategies")
        print("3. Clean separation eliminates code duplication")
        print("4. Backward compatibility maintained")
        print("5. Flexible configuration for different use cases")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed")
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if actual data is not available") 