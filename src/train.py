import torch
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

from ultralytics import YOLO
import yaml

def train_model():
    """Train YOLOv8 on power line defect dataset"""
    
    print("=" * 60)
    print("POWER LINE DEFECT DETECTOR - TRAINING")
    print("=" * 60)
    
    # Load pretrained model
    print("\nLoading YOLOv8n pretrained model...")
    model = YOLO('yolov8n.pt')
    
    # Check dataset config
    data= r"..\dataset\data.yaml"
    with open(data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    print("\nStarting training (will take 60-90 minutes)...\n")
    
    # Train
    results = model.train(
        data= data,
        epochs=50,
        imgsz=640,
        batch=16,
        name='powerline_detector',
        patience=10,
        device='cpu',  # Change to 'cuda' if you have GPU
        
        # Augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        
        # Save settings
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    best_model = results.save_dir / 'weights' / 'best.pt'
    print(f"\nBest model: {best_model}")
    
    return model, results


def validate_model(model_path='../runs/detect/powerline_detector/weights/best.pt'):
    """Validate the trained model"""
    
    print("\n" + "=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)
    
    model = YOLO(model_path)
    metrics = model.val()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"mAP50:     {metrics.box.map50:.3f} ({metrics.box.map50*100:.1f}%)")
    print(f"mAP50-95:  {metrics.box.map:.3f} ({metrics.box.map*100:.1f}%)")
    print(f"Precision: {metrics.box.mp:.3f} ({metrics.box.mp*100:.1f}%)")
    print(f"Recall:    {metrics.box.mr:.3f} ({metrics.box.mr*100:.1f}%)")
    print("=" * 60)
    
    if metrics.box.map50 >= 0.80:
        print("\n✅ Model achieved 80%+ mAP - READY FOR RESUME!")
    else:
        print(f"\n⚠️  Model at {metrics.box.map50*100:.1f}% mAP")
        print("Consider training longer")
    
    return metrics


if __name__ == "__main__":
    model, results = train_model()
    
    metrics = validate_model()
    
    print("\n✅ Next steps:")
    print("1. Check: runs/detect/powerline_detector/")
    print("2. Run: python evaluate.py")
    print("3. Run: python api.py")