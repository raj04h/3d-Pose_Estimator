import torch
import cv2
import numpy as np

from model_arch import poseNet

# trained model
trained_model= "SAT_Pose_model.pth"

# test img pth
test_img= r"D:\Data centr\IMG_data\satellite_pose\speed\images\test\img000092.jpg"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=poseNet()
model.load_state_dict(torch.load(trained_model, map_location=device))

model.to(device)
model.eval()

image = cv2.imread(test_img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224,224))
image = image / 255.0


# convert to tensor
img= torch.tensor(image).permute(2, 0, 1).float()
img=img.unsqueeze(0)
img=img.to(device)

with torch.no_grad():
    prediction= model(img)
pose=prediction.cpu().numpy()[0]

# split output
q=pose[:4]
t=pose[4:]

print("\nPredicted Quaternion (orientation):")
print(q)

print("\nPredicted Translation (position):")
print(t)