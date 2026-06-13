import importlib.util

import torch
from PIL import Image
from models.detector_interface import BaseDetector

class DetrDetector(BaseDetector):
    def __init__(self, model_id: str, device: str, confidence_threshold: float):
        """Initializes the DETR model.
        
        Args:
            model_id: Hugging Face model ID (e.g., 'facebook/detr-resnet-50').
            device: Device to load the model on ('cuda', 'cpu', etc.).
            confidence_threshold: Minimum confidence score to accept a detection.
        """
        self.model_id = model_id
        self.device = device
        self.confidence_threshold = confidence_threshold

        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "DETR requires the optional 'transformers' dependency. "
                "Install it to use '--model-type detr'."
            ) from exc

        if importlib.util.find_spec("timm") is None:
            raise ImportError(
                "DETR requires the optional 'timm' dependency for its image backbone. "
                "Install it to use '--model-type detr'."
            )
        
        # Load processor and model
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        self.model = transformers.AutoModelForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Image.Image) -> list:
        """Runs inference on a PIL Image and returns a list of standardized detections.
        
        Args:
            image: PIL Image to run inference on.
            
        Returns:
            A list of dicts with keys: 'label', 'score', 'box'.
        """
        width, height = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.confidence_threshold, 
            target_sizes=[(height, width)]
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.model.config.id2label[label.item()].lower()
            detections.append({
                "label": label_name,
                "score": score.item(),
                "box": box.tolist()  # [xmin, ymin, xmax, ymax]
            })
            
        return detections
