"""
Compare the model's inference accuracy using FP32 vs FP16 precision.
Expected Output: Print similarity or drift in predictions between FP32 and FP16 runs.
"""

import torch
import torchvision.models as models

model_fp32 = models.resnet18(pretrained=True).eval()
model_fp16 = models.resnet18(pretrained=True).half().eval()

input_fp32 = torch.randn(1, 3, 224, 224)
input_fp16 = input_fp32.half()

with torch.no_grad():
    output_fp32 = model_fp32(input_fp32)
    output_fp16 = model_fp16(input_fp16)

print("FP32 prediction:", output_fp32.argmax(dim=1).item())
print("FP16 prediction:", output_fp16.argmax(dim=1).item())
