from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from ultralytics import YOLO
import cv2
import numpy as np
import time
from metrics import metrics
import base64
import asyncio

MAX_INFLIGHT_INFERENCE = 9999  # Safety limit to prevent inference thrash under spikes ( when needed )
infer_semaphore = asyncio.Semaphore(MAX_INFLIGHT_INFERENCE)

app = FastAPI(
    title="Power Line Defect Detector API",
    description="Real-time defect detection for power line inspection",
    version="1.0.0"
)

# Load model
MODEL_PATH = '../runs/detect/powerline_detector/weights/best.pt'
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded!")

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

@app.get("/metrics", response_class=PlainTextResponse)
def metric_summary():
    return metrics.format_summary()

@app.post("/detect")
async def detect_defects(annotate: bool = True, file: UploadFile = File(...)):
    """Detect defects in uploaded image"""
    
    time_req0 = time.perf_counter()  # end to end request time

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        async with infer_semaphore:  # limit concurrent inferences
            time_model0 = time.perf_counter()  # model inference time start

            # Apply preprocessing
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Run inference
            results = model(preprocessed, conf=0.5, iou=0.45)

            # timing by 1000 to get milliseconds
            model_ms = (time.perf_counter() - time_model0) * 1000.0

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
            
            # Annotated image ( optional )
            img_base64 = None
            if annotate:
                annotated = results[0].plot()
                _, buffer = cv2.imencode('.jpg', annotated)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        fps = 1000.0 / model_ms if model_ms > 0 else 0
        
        total_ms = (time.perf_counter() - time_req0) * 1000.0
        metrics.record(endpoint="/detect", total_ms=total_ms, model_ms=model_ms, was_successful=True, items=1)

        response = {
            'success': True,
            'detections': detections,
            'num_defects': len(detections),
            'inference_time_ms': round(model_ms, 2),
            'fps': round(fps, 2),
        }
    
        if annotate:
            response["annotated_image"] = img_base64
        
        return response

    except HTTPException:   # using this one to actually get the timing even on bad requests
        total_ms = (time.perf_counter() - time_req0) * 1000.0
        metrics.record(endpoint="/detect", total_ms=total_ms, model_ms=None, was_successful=False, items=1)
        raise
    except Exception as e:
        total_ms = (time.perf_counter() - time_req0) * 1000.0
        metrics.record(endpoint="/detect", total_ms=total_ms, model_ms=None, was_successful=False, items=1)
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