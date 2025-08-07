"""
Export a trained PyTorch model to ONNX format for cross-platform and hardware-agnostic inference.
Expected Output: An ONNX model file (e.g., resnet18.onnx) saved to disk.
"""

import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, dummy_input, "resnet18.onnx", opset_version=11)
print("Exported model to resnet18.onnx")
