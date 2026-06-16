import torch
from app.model_arch import poseNet

model=poseNet()

dummy_input=torch.randn(1, 3, 224, 224)

output=model(dummy_input)

print("Output shape:", output.shape)
print("Pose prediction:", output) 