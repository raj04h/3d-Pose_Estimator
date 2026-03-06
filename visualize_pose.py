import torch
import cv2
import numpy as np
import plotly.graph_objects as go
import os

from model_arch import poseNet

model_path = "SAT_Pose_model.pth"

image_path = r"D:\Data centr\IMG_data\satellite_pose\speed\images\real_test\img000324real.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = poseNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

if not os.path.exists(image_path):
    raise ValueError("Image not found")

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(224,224))
image = image/255.0

img_tensor = torch.tensor(image).permute(2,0,1).float()
img_tensor = img_tensor.unsqueeze(0).to(device)

# INFERENCE
with torch.no_grad():
    pred = model(img_tensor)

pose = pred.cpu().numpy()[0]

q = pose[:4]
t = pose[4:]

print("\nQuaternion:", q)
print("Translation:", t)


# QUATERNION → ROTATION MATRIX
qw,qx,qy,qz = q

R = np.array([
[1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
[2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
[2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
])

# SATELLITE MODEL
cube = np.array([
[-1,-1,-1],
[1,-1,-1],
[1,1,-1],
[-1,1,-1],
[-1,-1,1],
[1,-1,1],
[1,1,1],
[-1,1,1]
])

edges = [
(0,1),(1,2),(2,3),(3,0),
(4,5),(5,6),(6,7),(7,4),
(0,4),(1,5),(2,6),(3,7)
]


# solar panels
panel_left = np.array([
[-3,0,0],
[-1,0,0]
])

panel_right = np.array([
[1,0,0],
[3,0,0]
])


# rotate + translate
cube = cube @ R.T + t

panel_left = panel_left @ R.T + t

panel_right = panel_right @ R.T + t


fig = go.Figure()

# DRAW SATELLITE BODY
for e in edges:

    p1 = cube[e[0]]
    p2 = cube[e[1]]

    fig.add_trace(go.Scatter3d(
        x=[p1[0],p2[0]],
        y=[p1[1],p2[1]],
        z=[p1[2],p2[2]],
        mode='lines',
        line=dict(color="cyan",width=6),
        showlegend=False
    ))

# SOLAR PANELS
fig.add_trace(go.Scatter3d(
x=panel_left[:,0],
y=panel_left[:,1],
z=panel_left[:,2],
mode='lines',
line=dict(color="orange",width=10),
name="Solar Panel"
))

fig.add_trace(go.Scatter3d(
x=panel_right[:,0],
y=panel_right[:,1],
z=panel_right[:,2],
mode='lines',
line=dict(color="orange",width=10),
showlegend=False
))

# SATELLITE AXES
origin = t

axes = np.eye(3)

colors = ['red','green','blue']

labels = ["X axis","Y axis","Z axis"]

for i in range(3):

    axis = axes[i] @ R.T

    fig.add_trace(go.Scatter3d(
        x=[origin[0], origin[0]+axis[0]],
        y=[origin[1], origin[1]+axis[1]],
        z=[origin[2], origin[2]+axis[2]],
        mode='lines',
        line=dict(color=colors[i],width=8),
        name=labels[i]
    ))


# CAMERA AXES
camera_origin = np.array([0,0,0])

camera_axes = np.eye(3)

colors = ['darkred','darkgreen','darkblue']

for i in range(3):

    fig.add_trace(go.Scatter3d(
        x=[camera_origin[0],camera_axes[i][0]*2],
        y=[camera_origin[1],camera_axes[i][1]*2],
        z=[camera_origin[2],camera_axes[i][2]*2],
        mode='lines',
        line=dict(color=colors[i],width=6),
        name="Camera Axis" if i==0 else None
    ))


# Layout
fig.update_layout(

title="Satellite Pose Visualization",

scene=dict(
    xaxis_title="X",
    yaxis_title="Y",
    zaxis_title="Z",
    aspectmode="data"
),

paper_bgcolor="white",
font=dict(color="black")

)

fig.show()