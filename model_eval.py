import torch
import json
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model_arch import poseNet


trained_model = "SAT_Pose_model.pth"

image_folder = r"D:\Data centr\IMG_data\satellite_pose\speed\images\train"
json_path = r"D:\Data centr\IMG_data\satellite_pose\speed\train.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = poseNet()
model.load_state_dict(torch.load(trained_model, map_location=device))
model.to(device)
model.eval()


# load labels
with open(json_path) as f:
    data = json.load(f)

rotation_errors = []
translation_errors = []

pred_positions = []
gt_positions = []


# Evaluation loop
for sample in tqdm(data):

    filename = sample["filename"]
    img_pth = os.path.join(image_folder, filename)
    image = cv2.imread(img_pth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0

    img_tensor = torch.tensor(image).permute(2,0,1).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)


    with torch.no_grad():
        pred = model(img_tensor)

    pred = pred.cpu().numpy()[0]


    q_pred = pred[:4]
    t_pred = pred[4:]

    q_gt = np.array(sample["q_vbs2tango"])
    t_gt = np.array(sample["r_Vo2To_vbs_true"])

    # rotation error
    dot = np.abs(np.dot(q_pred, q_gt))
    dot = np.clip(dot, -1.0, 1.0)
    rot_error = 2 * np.arccos(dot)

    rotation_errors.append(rot_error)

    # translation error
    trans_error = np.linalg.norm(t_pred - t_gt)
    translation_errors.append(trans_error)

    pred_positions.append(t_pred)
    gt_positions.append(t_gt)


rotation_errors = np.array(rotation_errors)
translation_errors = np.array(translation_errors)

pred_positions = np.array(pred_positions)
gt_positions = np.array(gt_positions)


print("\nAverage Rotation Error:", rotation_errors.mean())
print("Average Translation Error:", translation_errors.mean())


# plot rotation error histogram
plt.figure(figsize=(8,6))
plt.hist(rotation_errors, bins=40, color="blue")
plt.title("Rotation Error Distribution")


plt.xlabel("Rotation Error (radians)")
plt.ylabel("Frequency")


plt.grid(True)
plt.savefig(r"Evaluation_Graph\rotation_error.jpg")
plt.close()

# plot translation error histogram
plt.figure(figsize=(8,6))
plt.hist(translation_errors, bins=40, color="green")
plt.title("Translation Error Distribution")

plt.xlabel("Translation Error (meters)")
plt.ylabel("Frequency")

plt.grid(True)
plt.savefig(r"Evaluation_Graph\translation_error.jpg")
plt.close()

# plot rotation vs translation error
plt.figure(figsize=(8,6))
plt.scatter(rotation_errors, translation_errors)
plt.title("Rotation vs Translation Error")


plt.xlabel("Rotation Error")
plt.ylabel("Translation Error")

plt.grid(True)
plt.savefig(r"Evaluation_Graph\pose_error_scatter.jpg")
plt.close()


# predicted vs ground truth positions
plt.figure(figsize=(8,6))
plt.scatter(gt_positions[:,2], pred_positions[:,2])


plt.xlabel("Ground Truth Distance (Z)")
plt.ylabel("Predicted Distance (Z)")

plt.title("Predicted vs Ground Truth Distance")
plt.grid(True)
plt.savefig(r"Evaluation_Graph\distance_comparison.jpg")
plt.close()

print("\nEvaluation graphs plotted successfully")