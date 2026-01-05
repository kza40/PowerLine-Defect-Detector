from ultralytics import YOLO
from pathlib import Path
import cv2

def evaluate_model(model_path='../runs/detect/powerline_detector/weights/best.pt'):
    """Comprehensive model evaluation"""
    
    print("Loading model...")
    model = YOLO(model_path)
    
    print("Running validation...")
    metrics = model.val()
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    print("=" * 60)
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    print("-" * 60)
    class_names = model.names
    for i, class_name in class_names.items():
        print(f"{class_name:20s} - mAP50: {metrics.box.maps[i]:.3f}")
    
    return metrics


def test_on_samples(model_path='../runs/detect/powerline_detector/weights/best.pt', num=10):
    """Test on sample images"""
    
    print("\nTesting on sample images...")
    model = YOLO(model_path)
    
    test_images = list(Path('dataset/test/images').glob('*.jpg'))[:num]
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    output_dir = Path('../runs/detect/test_predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, img_path in enumerate(test_images):
        print(f"Processing {idx+1}/{len(test_images)}: {img_path.name}")
        
        results = model(str(img_path), conf=0.5)
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
        
        # Print detections
        detections = results[0].boxes
        print(f"  Found {len(detections)} objects")
        for box in detections:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"    - {cls_name}: {conf:.2f}")
    
    print(f"\n✅ Saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Evaluate
    metrics = evaluate_model()
    
    # Test samples
    test_on_samples(num=10)
    
    print("\n✅ COMPLETE!")
    print(f"\nResult: Achieved {metrics.box.map50*100:.0f}%+ mAP on 1,000+ images")