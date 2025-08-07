"""
Profile the training/inference steps of a ResNet model using PyTorch Profiler.
Expected Output: Print or save performance metrics, especially around GPU/CPU time per operation.
"""

import torch
import torchvision.models as models
import torch.profiler

model = models.resnet18(pretrained=True)
input_tensor = torch.randn(1, 3, 224, 224)

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=True
) as prof:
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_tensor)
        prof.step()

print("Profiler output saved to ./log directory (for TensorBoard)")
