from abc import ABC, abstractmethod
from PIL import Image

class BaseDetector(ABC):
    @abstractmethod
    def __init__(self, model_id: str, device: str, confidence_threshold: float):
        """Initializes the model, loads weights, and places it on the device.
        
        Args:
            model_id: Hugging Face model ID or model path.
            device: Execution device ('cuda', 'cpu', etc.).
            confidence_threshold: Threshold to filter out detections.
        """
        pass

    @abstractmethod
    def detect(self, image: Image.Image) -> list:
        """Runs object detection on a PIL Image.
        
        Args:
            image: A PIL Image to analyze.
            
        Returns:
            A list of dictionaries representing detections:
            [
                {
                    "label": str,       # Class label (e.g. 'person', 'car', 'dog')
                    "score": float,     # Confidence score (0.0 to 1.0)
                    "box": [xmin, ymin, xmax, ymax] # Bounding box coordinates
                },
                ...
            ]
        """
        pass
