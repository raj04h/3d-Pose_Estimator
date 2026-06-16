import torch
import torch.nn as nn
import torchvision.models as model
from torchvision.models import resnet18, ResNet18_Weights

class poseNet(nn.Module):
    def __init__(self):
        super(poseNet, self).__init__()

        # Load pretrained ResNet18 model extract feature from image
        # eg- solar panels, body edge, lighting reflection

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Original ResNet outputs 1000 classes
        self.backbone.fc=nn.Linear(512, 7)

    def forward(self, x):
        output=self.backbone(x)

        q= output[:, :4]  # First 4 values represent quaternion rotation
        t=output[:, 4:] # Last 3 values represent translation

        q=q/torch.norm(q, dim=1, keepdim=True)  # Normalize,so its magnitude = 1
        pose=torch.cat((q,t), dim=1)

        return pose