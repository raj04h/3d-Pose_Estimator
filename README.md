# Satellite Pose Estimation using Vision AI

Overview

This project implements an AI-based satellite pose estimation system using computer vision and deep learning.
Given a monocular image of a satellite, the model predicts its 6D pose, which includes:

* Orientation (Quaternion) → (qw, qx, qy, qz)

* Translation (Position) → (x, y, z)

The predicted pose is then visualized in 3D space to demonstrate how the satellite is oriented relative to the observing camera.

This type of perception system is important for space robotics tasks such as:

* On-orbit servicing

* Autonomous rendezvous and docking

* Space debris removal

* Satellite inspection missions

# Model Result-

* Web App-
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cf7ac309-3495-4aea-9372-6e6d0582524f" />

* Sample Image-
![img000195real](https://github.com/user-attachments/assets/26ebe99e-9d0b-4f04-9410-81f09ddd0e28)

* 3d Image-
![3d_visualization](https://github.com/user-attachments/assets/86812dc9-8ab0-46ec-8b0c-548e560eadd4)

![distance_comparison](https://github.com/user-attachments/assets/e201ca9f-8264-4445-815c-e20e2b288183)

![pose_error_scatter](https://github.com/user-attachments/assets/ee67648f-298b-4e3e-90c8-c9ad44f7c072)

<img width="800" height="500" alt="Training_Loss" src="https://github.com/user-attachments/assets/594e73f4-ac67-4279-a0fc-bbf04713f5d6" />

# Project Pipeline

Satellite Image

↓

Deep Learning Model (ResNet18 PoseNet)

↓

Pose Regression (Quaternion + Translation)

↓

Model Evaluation

↓

3D Satellite Pose Visualization

↓

Interactive Pose Viewer

# Technologies Used

* Python

* PyTorch

* OpenCV

* NumPy

* Matplotlib

* Plotly

# Dataset

The project uses the SPEED (Satellite Pose Estimation Dataset).

Pose labels include:

* q_vbs2tango   → quaternion orientation
* r_Vo2To_vbs_true → translation vector

# Model Architecture

The pose estimation model is implemented using PyTorch.
Quaternion normalization is applied to ensure valid rotation representation.

* Architecture:

Input Image (224x224)

↓

ResNet18 Backbone

↓

Fully Connected Layer

↓

7D Pose Vector

* Output:

[qw qx qy qz tx ty tz]

* Where:
qw qx qy qz → rotation quaternion

tx ty tz → translation vector

# Model Training & Evaluation

Train the pose estimation model:

* SAT_Pose_model.pth
* Training_Loss.png
Generated metrics:

* Rotation error distribution

* Translation error distribution

* Pose error scatter plot

# 3D Pose Visualization:
The visualization shows:

* Satellite body

* Solar panels

* Camera coordinate frame

* Satellite coordinate frame

* Estimated translation

# Interactive Visualization

Features:

* Rotate satellite with mouse

* Zoom and pan

* Inspect satellite orientation

* View coordinate axes

# Future Improvements

* Transformer-based pose estimation

* Real-time onboard inference

* Multi-view pose estimation

* Integration with spacecraft navigation systems

# Author- Himanshu Raj
© 2026, All rights reserved.

