import streamlit as st
import torch
import cv2
import numpy as np
import plotly.graph_objects as go
import sys
import os

# allow importing model from parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model_arch import poseNet


# PAGE CONFIG
st.set_page_config(
    page_title="Satellite Pose Estimation",
    page_icon=" ",
    layout="wide"
)

st.title("Satellite Pose Estimation Dashboard")

st.markdown(
"""
Upload a **satellite image** and the AI model will estimate its **6DoF pose**
(orientation + translation) and visualize the spacecraft in **3D space**.
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



# UTILITY FUNCTIONS
def quaternion_to_rotation(q):

    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

    return R


def create_satellite_plot(R, t):

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

    panel_left = np.array([[-3,0,0],[-1,0,0]])
    panel_right = np.array([[1,0,0],[3,0,0]])

    cube = cube @ R.T + t
    panel_left = panel_left @ R.T + t
    panel_right = panel_right @ R.T + t

    fig = go.Figure()

    # satellite body
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

    # solar panels
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

    # satellite coordinate axes
    axes = np.eye(3)
    colors = ['red','green','blue']
    labels = ["X axis","Y axis","Z axis"]

    for i in range(3):

        axis = axes[i] @ R.T

        fig.add_trace(go.Scatter3d(
            x=[t[0],t[0]+axis[0]],
            y=[t[1],t[1]+axis[1]],
            z=[t[2],t[2]+axis[2]],
            mode='lines',
            line=dict(color=colors[i],width=8),
            name=labels[i]
        ))

    # camera frame
    camera_axes = np.eye(3)

    for i in range(3):

        fig.add_trace(go.Scatter3d(
            x=[0,camera_axes[i][0]*2],
            y=[0,camera_axes[i][1]*2],
            z=[0,camera_axes[i][2]*2],
            mode='lines',
            line=dict(color="gray",width=4),
            showlegend=False
        ))

    fig.update_layout(

        title="3D Satellite Pose Visualization",

        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),

        paper_bgcolor="black"
    )

    return fig


# IMAGE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg","jpeg","png"]
)


if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes,1)

    col1, col2 = st.columns([1,1.3])

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    # preprocess
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = img/255.0

    img_tensor = torch.tensor(img).permute(2,0,1).float()
    img_tensor = img_tensor.unsqueeze(0)

    # inference
    with torch.no_grad():

        pred = model(img_tensor)

    pose = pred.numpy()[0]

    q = pose[:4]
    t = pose[4:]

    with col2:

        st.subheader("Predicted Pose")

        st.write("**Quaternion (Orientation)**")
        st.code(q)

        st.write("**Translation (Position)**")
        st.code(t)

        R = quaternion_to_rotation(q)

        fig = create_satellite_plot(R,t)

        st.plotly_chart(fig, use_container_width=True)