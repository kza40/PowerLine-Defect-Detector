from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import time
from pathlib import Path
import tempfile
import base64
from preprocess import preprocess_image

app = FastAPI(
    title="Power Line Defect Detector API",
    description="Real-time defect detection for power line inspection",
    version="1.0.0"
)

# Load model
MODEL_PATH = '../runs/detect/powerline_detector/weights/best.pt'
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded!")

@app.get("/")
async def root():
    return {
        "status": "online",
        "model": "YOLOv8n Power Line Defect Detector",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": list(model.names.values())
    }

@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)):
    """Detect defects in uploaded image"""
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Preprocess directly without temp file
        start_time = time.time()
        
        # Apply preprocessing
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Run inference
        results = model(preprocessed, conf=0.5, iou=0.45)
        inference_time = time.time() - start_time
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': {
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3])
                }
            })
        
        # Annotated image
        annotated = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        fps = 1 / inference_time if inference_time > 0 else 0
        
        return {
            'success': True,
            'detections': detections,
            'num_defects': len(detections),
            'inference_time_ms': round(inference_time * 1000, 2),
            'fps': round(fps, 2),
            'annotated_image_base64': img_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Starting API Server")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)