"""
Example demonstrating the CLIP-based zero-shot activity classifier.

This example shows how to use CLIP models for efficient zero-shot activity
recognition without requiring fine-tuning.
"""

import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_clip_usage():
    """Example of basic CLIP classifier usage."""
    print("=== Basic CLIP Classifier Usage ===")
    
    from src.watcher.activity.clip_classifier import create_clip_classifier
    from src.watcher.activity.base import TrackedObject, ActivityType
    
    # Create CLIP classifier with default settings
    clip_classifier = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        device="auto",
        confidence_threshold=0.6,
        ambiguous_threshold=0.3
    )
    
    print(f"Created CLIP classifier: {type(clip_classifier).__name__}")
    print(f"Model info: {clip_classifier.get_model_info()}")
    
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
    
    # Example usage (commented out since we don't have actual image data)
    # import numpy as np
    # 
    # # Simulate some image data
    # images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    # 
    # # Classify activity
    # result = clip_classifier(images, tracked_obj)
    # print(f"Activity result: {result.activity_type.value}")
    # print(f"Confidence: {result.confidence:.3f}")


def example_different_clip_models():
    """Example of using different CLIP model variants."""
    print("\n=== Different CLIP Model Variants ===")
    
    from src.watcher.activity.clip_classifier import create_clip_classifier
    
    # 1. Standard CLIP
    print("\n--- Standard CLIP ---")
    standard_clip = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        model_type="clip"
    )
    print(f"Standard CLIP: {standard_clip.get_model_info()['model_type']}")
    
    # 2. SigLIP (Google's improved CLIP)
    print("\n--- SigLIP ---")
    siglip_clip = create_clip_classifier(
        model_path="google/siglip-base-patch16-224",
        model_type="siglip"
    )
    print(f"SigLIP: {siglip_clip.get_model_info()['model_type']}")
    
    # 3. OpenCLIP
    print("\n--- OpenCLIP ---")
    openclip_classifier = create_clip_classifier(
        model_path="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        model_type="openclip"
    )
    print(f"OpenCLIP: {openclip_classifier.get_model_info()['model_type']}")


def example_temporal_aggregation():
    """Example of different temporal aggregation methods."""
    print("\n=== Temporal Aggregation Methods ===")
    
    from src.watcher.activity.clip_classifier import create_clip_classifier
    
    # 1. Mean aggregation (default)
    mean_clip = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        use_temporal_aggregation=True,
        aggregation_method="mean"
    )
    print(f"Mean aggregation: {mean_clip.aggregation_method}")
    
    # 2. Max aggregation
    max_clip = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        use_temporal_aggregation=True,
        aggregation_method="max"
    )
    print(f"Max aggregation: {max_clip.aggregation_method}")
    
    # 3. Weighted aggregation
    weighted_clip = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        use_temporal_aggregation=True,
        aggregation_method="weighted"
    )
    print(f"Weighted aggregation: {weighted_clip.aggregation_method}")
    
    # 4. No temporal aggregation
    no_agg_clip = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        use_temporal_aggregation=False
    )
    print(f"No temporal aggregation: {no_agg_clip.use_temporal_aggregation}")


def example_activity_labels():
    """Example showing the activity labels used for zero-shot classification."""
    print("\n=== Activity Labels for Zero-Shot Classification ===")
    
    from src.watcher.activity.clip_classifier import create_clip_classifier
    
    clip_classifier = create_clip_classifier()
    
    # Get the activity labels that CLIP will use
    labels = clip_classifier._generate_activity_labels()
    
    print("CLIP will classify images using these labels:")
    for i, label in enumerate(labels, 1):
        print(f"{i:2d}. {label}")
    
    print(f"\nTotal labels: {len(labels)}")


def example_integration_with_recognizer():
    """Example of integrating CLIP classifier with the activity recognizer."""
    print("\n=== Integration with Activity Recognizer ===")
    
    from src.watcher.activity.clip_classifier import create_clip_classifier
    from src.watcher.activity.recognizer import DroneActivityRecognizer
    
    # Create CLIP classifier
    clip_classifier = create_clip_classifier(
        model_path="openai/clip-vit-base-patch32",
        confidence_threshold=0.6,
        ambiguous_threshold=0.3
    )
    
    # Create activity recognizer with CLIP classifier
    activity_recognizer = DroneActivityRecognizer(
        activity_classifier=clip_classifier,
        sequence_length=8,
        input_size=(224, 224)
    )
    
    print("Created activity recognizer with CLIP classifier")
    print("This provides:")
    print("- Zero-shot activity classification")
    print("- No fine-tuning required")
    print("- Fast inference")
    print("- Temporal aggregation across frames")


def example_performance_comparison():
    """Example comparing CLIP with other activity recognition approaches."""
    print("\n=== Performance Comparison ===")
    
    print("CLIP Classifier Advantages:")
    print("✓ Zero-shot classification (no fine-tuning needed)")
    print("✓ Fast inference")
    print("✓ Good generalization")
    print("✓ Multiple model variants available")
    print("✓ Temporal aggregation support")
    
    print("\nCLIP Classifier Limitations:")
    print("✗ May not be as accurate as fine-tuned models")
    print("✗ Limited to predefined activity labels")
    print("✗ Requires good text descriptions")
    
    print("\nUse Cases:")
    print("- Quick prototyping")
    print("- General activity detection")
    print("- When fine-tuning data is limited")
    print("- Real-time applications")


def example_custom_labels():
    """Example of using custom activity labels."""
    print("\n=== Custom Activity Labels ===")
    
    from src.watcher.activity.clip_classifier import CLIPClassifier
    from src.watcher.activity.base import ActivityType
    
    # Create custom activity prompts
    custom_prompts = {
        ActivityType.NORMAL_WALKING: "a person walking slowly and casually",
        ActivityType.RUNNING: "a person running fast or sprinting",
        ActivityType.FIGHTING: "two or more people fighting or brawling",
        ActivityType.GROUP_GATHERING: "multiple people standing in a group",
        ActivityType.MILITIA_BEHAVIOR: "people in uniform or military formation",
        ActivityType.VEHICLE_DRIVING: "a car or truck moving on the road",
        ActivityType.VEHICLE_SPEEDING: "a vehicle moving very fast",
        ActivityType.VEHICLE_STOPPING: "a vehicle parked or stopped",
        ActivityType.SUSPICIOUS_BEHAVIOR: "someone acting suspiciously or hiding",
        ActivityType.CROWD_FORMATION: "a large group of people moving together",
        ActivityType.UNKNOWN: "nothing happening or unclear activity"
    }
    
    # Create CLIP classifier with custom prompts
    custom_clip = CLIPClassifier(
        model_path="openai/clip-vit-base-patch32",
        confidence_threshold=0.6
    )
    custom_clip.activity_prompts = custom_prompts
    
    print("Created CLIP classifier with custom activity labels")
    print("Custom labels provide more specific descriptions for better classification")


if __name__ == "__main__":
    print("CLIP Classifier Examples")
    print("=" * 50)
    
    try:
        example_basic_clip_usage()
        example_different_clip_models()
        example_temporal_aggregation()
        example_activity_labels()
        example_integration_with_recognizer()
        example_performance_comparison()
        example_custom_labels()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey Benefits of CLIP Classifier:")
        print("1. Zero-shot classification - no training required")
        print("2. Fast inference with good accuracy")
        print("3. Multiple model variants (CLIP, SigLIP, OpenCLIP)")
        print("4. Temporal aggregation for video sequences")
        print("5. Customizable activity labels")
        print("6. Easy integration with existing pipeline")
        
        print("\nTo use with actual data:")
        print("1. Install transformers: pip install transformers")
        print("2. Load your image/video data")
        print("3. Create tracked objects")
        print("4. Call the classifier with your data")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required dependencies: pip install transformers")
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if actual data is not available") 