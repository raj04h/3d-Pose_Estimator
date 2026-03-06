import streamlit as st
import torch
import cv2
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_arch import poseNet


# PAGE CONFIG
st.set_page_config(page_title="Satellite Pose Estimation", layout="wide")
st.title("Satellite Pose Estimation System")

st.write(
"""
Upload a satellite image and the AI model will estimate the **6DoF pose**
(orientation + translation) and visualize the satellite in **3D space**.
"""
)

# LOAD MODEL
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = poseNet()
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "SAT_Pose_model.pth"
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# IMAGE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="Input Image", use_container_width=True)

    # PREPROCESS IMAGE
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)

    # MODEL INFERENCE
    with torch.no_grad():
        pred = model(img_tensor)

    pose = pred.numpy()[0]
    q = pose[:4]
    t = pose[4:]

    st.subheader("Predicted Pose")
    col1, col2 = st.columns(2)
    col1.write("**Quaternion (Orientation)**")
    col1.write(q)

    col2.write("**Translation (Position)**")
    col2.write(t)

    # QUATERNION → ROTATION MATRIX
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

    # SATELLITE MODEL
    cube = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ])

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]


    cube = cube @ R.T + t
    fig = go.Figure()

   # DRAW CUBE
    for e in edges:
        p1 = cube[e[0]]
        p2 = cube[e[1]]

        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]],
            y=[p1[1], p2[1]],
            z=[p1[2], p2[2]],
            mode='lines',
            line=dict(color="cyan", width=6),
            showlegend=False
        ))

    # SATELLITE AXES
    origin = t

    axes = np.eye(3)

    colors = ['red', 'green', 'blue']

    labels = ["X axis", "Y axis", "Z axis"]


    for i in range(3):

        axis = axes[i] @ R.T

        fig.add_trace(go.Scatter3d(
            x=[origin[0], origin[0] + axis[0]],
            y=[origin[1], origin[1] + axis[1]],
            z=[origin[2], origin[2] + axis[2]],
            mode='lines',
            line=dict(color=colors[i], width=8),
            name=labels[i]
        ))

    # LAYOUT
    fig.update_layout(

        title="3D Satellite Pose",

        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        )
    )


    st.plotly_chart(fig, use_container_width=True)