import cv2
import numpy as np
from typing import List, Tuple, Union, Any
from ultralytics import YOLO
from app.core.logger_utils import CustomLogger

class YOLOPredictor:
    def __init__(self, model_name: str = 'yolov8n.pt') -> None:
        self.logger: CustomLogger = CustomLogger("yolo_predictor")
        self.model_name: str = model_name
        try:
            self.logger.info(f"Loading YOLO model: {self.model_name}")
            self.model: YOLO = YOLO(self.model_name)
            self.logger.info("YOLO model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def get_cropped_predictions(self, image_input: Union[str, np.ndarray]) -> List[Tuple[np.ndarray, List[int]]]:
        self.logger.info("Starting YOLO prediction.")
        try:
            results: Any = self.model(image_input)
            detections: List[Tuple[np.ndarray, List[int]]] = []

            if isinstance(image_input, str):
                img = cv2.imread(image_input)
            else:
                img = image_input

            if img is None:
                self.logger.error("Could not read image for cropping.")
                raise ValueError("Could not read image.")

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    detections.append((crop, [x1, y1, x2, y2]))
            
            self.logger.info(f"YOLO prediction finished. Detections: {len(detections)}")
            return detections
        except Exception as e:
            self.logger.error(f"Error during YOLO prediction: {e}")
            return []
