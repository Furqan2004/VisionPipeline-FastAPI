import cv2
import numpy as np
import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from app.api.dependencies import get_models
from app.core.config import settings
from app.core.logger_utils import CustomLogger

router = APIRouter()
logger = CustomLogger("api")

def apply_segmentation(frame, mask, bbox, color=(255, 120, 30)):
    x1, y1, x2, y2 = bbox
    h_crop, w_crop = mask.shape
    roi = frame[y1:y1+h_crop, x1:x1+w_crop]
    
    if roi.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (roi.shape[1], roi.shape[0])).astype(bool)

    overlay = roi.copy()
    overlay[mask] = (roi[mask] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
    frame[y1:y1+h_crop, x1:x1+w_crop] = overlay
    return frame

@router.get("/health")
async def health_check(models=Depends(get_models)):
    yolo, sam, classifier = models
    return {
        "status": "healthy",
        "models": {
            "yolo": "loaded" if yolo else "failed",
            "sam": "loaded" if sam else "failed",
            "classifier": "loaded" if classifier else "failed"
        }
    }

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...), models=Depends(get_models)):
    yolo, sam, classifier = models
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        detections = yolo.get_cropped_predictions(frame)
        results = []
        viz_frame = frame.copy()

        for i, (crop, bbox) in enumerate(detections):
            mask = sam.get_head_segmentation(crop)
            label = classifier.predict(crop)
            
            viz_frame = apply_segmentation(viz_frame, mask, bbox)
            cv2.rectangle(viz_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(viz_frame, f"ID:{i} L:{label}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            results.append({"id": i, "label": label, "bbox": bbox})
        
        out_dir = settings.get("output_dir", "output")
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        out_name = f"res_{uuid.uuid4().hex[:8]}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, viz_frame)

        return {"labels": results, "output_image": out_path}
    except Exception as e:
        logger.error(f"Image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-video")
async def predict_video(file: UploadFile = File(...), models=Depends(get_models)):
    yolo, sam, classifier = models
    ext = os.path.splitext(file.filename)[1]
    temp_input = f"temp_v_{uuid.uuid4().hex[:8]}{ext}"
    
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        cap = cv2.VideoCapture(temp_input)
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        
        out_dir = settings.get("output_dir", "output")
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        out_name = f"res_{uuid.uuid4().hex[:8]}.avi"
        out_path = os.path.join(out_dir, out_name)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        interval = settings.video_config.get("frame_interval", 1)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % interval == 0:
                detections = yolo.get_cropped_predictions(frame)
                for i, (crop, bbox) in enumerate(detections):
                    mask = sam.get_head_segmentation(crop)
                    label = classifier.predict(crop)
                    frame = apply_segmentation(frame, mask, bbox)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"L:{label}", (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        return FileResponse(out_path, media_type='video/x-msvideo', filename=out_name)
    finally:
        if os.path.exists(temp_input): os.remove(temp_input)
