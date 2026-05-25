from PIL import Image
from ultralytics import YOLO
from models.detector_interface import BaseDetector

class YoloDetector(BaseDetector):
    def __init__(self, model_id: str, device: str, confidence_threshold: float):
        """Initializes the YOLOv8 model.
        
        Args:
            model_id: YOLOv8 model file name or ID (e.g. 'yolov8n.pt').
            device: Device to load the model on ('cuda', 'cpu', etc.).
            confidence_threshold: Minimum confidence score to accept a detection.
        """
        self.model_id = model_id
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 model
        self.model = YOLO(model_id)
        # Move to target device
        self.model.to(device)

    def detect(self, image: Image.Image) -> list:
        """Runs YOLOv8 inference on a PIL Image and returns standardized detections.
        
        Args:
            image: PIL Image to run inference on.
            
        Returns:
            A list of dicts with keys: 'label', 'score', 'box'.
        """
        # YOLOv8 can process PIL Images directly
        results = self.model(image, conf=self.confidence_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            label_name = results.names[cls_id].lower()
            score = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            
            detections.append({
                "label": label_name,
                "score": score,
                "box": xyxy
            })
            
        return detections
