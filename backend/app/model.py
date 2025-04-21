from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

# Load model and feature extractor (on first import)
extractor = ViTFeatureExtractor.from_pretrained("amaye15/ViT-Standford-Dogs")
model = ViTForImageClassification.from_pretrained("amaye15/ViT-Standford-Dogs")

# Load breed labels from model config
id2label = model.config.id2label

def predict_breed(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(3)

    results = []
    for i in range(top_probs.size(1)):
        breed = id2label[top_indices[0][i].item()]
        confidence = top_probs[0][i].item()
        results.append({"breed": breed, "confidence": round(confidence, 3)})

    return results