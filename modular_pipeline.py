"""
Run a modular, production-style inference pipeline.
Expected Output: Inference results printed from a structured pipeline using reusable components.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

def load_model():
    model = resnet18(pretrained=True)
    model.eval()
    return model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        return output.argmax(dim=1).item()

if __name__ == "__main__":
    model = load_model()
    input_tensor = preprocess("sample.jpg")  # Replace with an actual image path
    prediction = predict(model, input_tensor)
    print(f"Predicted class index: {prediction}")
