from app.predictors.yolo_predictor import YOLOPredictor
from app.predictors.sam_predictor import SAM2Predictor
from app.predictors.classifier import EfficientNetClassifier
from app.core.config import settings

yolo = None
sam = None
classifier = None

def get_models():
    global yolo, sam, classifier
    if yolo is None:
        yolo = YOLOPredictor(model_name=settings.yolo_model_name)
    if sam is None:
        sam = SAM2Predictor(
            config_file=settings.sam_config_file,
            ckpt_path=settings.sam_ckpt_path
        )
    if classifier is None:
        classifier = EfficientNetClassifier(
            model_path=settings.model_path,
            image_size=settings.image_size,
            threshold=settings.threshold
        )
    return yolo, sam, classifier
