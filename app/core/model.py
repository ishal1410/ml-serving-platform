import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import boto3
import io
import os
from app.core.config import settings

# ImageNet class labels (top 10 for demo)
LABELS = [
    "tench", "goldfish", "great white shark", "tiger shark",
    "hammerhead shark", "electric ray", "stingray", "cock", "hen", "ostrich"
]

class ModelManager:
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_from_s3(self):
        """Load model from AWS S3"""
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            buffer = io.BytesIO()
            s3.download_fileobj(settings.S3_BUCKET, settings.MODEL_KEY, buffer)
            buffer.seek(0)
            self.model = torch.load(buffer, map_location=torch.device('cpu'))
            self.model.eval()
            print("Model loaded from S3")
        except Exception as e:
            print(f"S3 load failed: {e}. Loading default model...")
            self.load_default()

    def load_default(self):
        """Load pretrained EfficientNet as default"""
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.eval()
        print("Default EfficientNet-B0 model loaded")

    def predict(self, image: Image.Image) -> dict:
        """Run inference on image"""
        if self.model is None:
            self.load_default()

        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5 = torch.topk(probabilities, 5)

        results = []
        for prob, idx in zip(top5.values, top5.indices):
            label = LABELS[idx] if idx < len(LABELS) else f"class_{idx}"
            results.append({
                "label": label,
                "confidence": round(prob.item() * 100, 2)
            })
        return {"predictions": results}

model_manager = ModelManager()