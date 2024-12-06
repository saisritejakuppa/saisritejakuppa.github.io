---
layout: post
title:  Introduction to Pybullet
date:   2022-09-22 16:40:16
description: Pybullet tutorial, Load and View robots.
tags: Robotics
categories: Pybullet
---
# Pybullet: Robots and Cameras
Welcome to the tutorial on using PyBullet, the physics engine for simulating rigid body dynamics. In this blog post, we will be diving into the basics of PyBullet and how to use it to simulate physics in your own projects. Whether you're a beginner or an experienced developer, this tutorial will provide you with the knowledge and tools you need to get started with PyBullet. So, let's get started and learn how to create realistic physics simulations with PyBullet!

<p align="center">
<img src="https://user-images.githubusercontent.com/70435083/215378837-13b4e1a8-f2db-4aff-96ff-e76ea7639b80.gif" alt="ultrasoundgif" class="center">
</p>

<p align="center">
<a href="">UR5 robot interaction with a torid soft body.</a>
</p>


In this blog, we will be utilizing a UR5 robot to demonstrate the capabilities of PyBullet. We will guide you through the process of loading the robot into the simulation and show you how to manipulate its movement. Additionally, we will also explore the use of cameras to view the robot from different angles, providing a more realistic representation of its motion


URDF (Unified Robot Description Format) is a file format used to describe the physical structure and kinematics of a robot. It is an XML-based format that is used to define the robot's links, joints, and sensors. The URDF file contains information about the robot's geometry, mass properties, joint limits, and other parameters. It is used to define the robot's model for physics simulation and visualization.

URDF is widely used in robotics, it is used in popular robot simulators like Gazebo and PyBullet, as well as in many robot operating systems like ROS. URDF files can be created manually or generated from CAD models using various tools like xacro, URDF exporter from SolidWorks, Inventor, etc. The URDF file can be loaded into a robot simulator, and the robot's model can be used for physics simulation, motion planning, and visualization.



PyBullet, a physics simulation engine, also provides support for simulating soft-body dynamics. Soft-bodies are objects that can bend, stretch, and deform, unlike rigid bodies that maintain a fixed shape. Examples of soft-bodies include fabrics, ropes, and other flexible materials.

In PyBullet, soft-bodies are represented using the Bullet Soft Body library. This library allows users to create and simulate soft-bodies using a variety of methods, including cloth, rope, and mesh simulations. The library also provides several parameters that can be used to control the behavior of the soft-body, such as stiffness, damping, and mass.

The user can create a soft-body using the p.createSoftBody function. This function takes several arguments such as the shape of the soft-body, its mass, and its collision shape. After creating the soft-body, the user can apply forces and torques to it, and the library will simulate its dynamics accordingly.

One of the main advantages of using soft-bodies in PyBullet is the ability to create realistic simulations of flexible materials such as fabrics, ropes, and other flexible materials. This can be useful in many applications such as robotics, animation, and gaming.

<p align="center">
<img src="https://user-images.githubusercontent.com/70435083/215378959-46fa9d30-1bc1-4625-805b-d64a319fbf57.gif" alt="ultrasoundgif" class="center">
</p>

<p align="center">
<a href="">Interaction with the soft body.</a>
</p>




<p align="center">
<img src="https://user-images.githubusercontent.com/70435083/215380122-689f9c61-64c1-46ee-941c-426424f31743.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Close view of robot interacting with the toroid.</a>
</p>



python code
  
## Install the dependencies and load the necessary headers

```python
#install the dependencies and load the necessary headers
import pybullet as p
import pybullet_data
import time

#connect to the physics server
p.connect(p.DIRECT)
#allow to find the assets (URDF, obj, textures etc)
p.setAdditionalSearchPath(pd.getDataPath())

p.setGravity(0,0,-10)

#load the ground plane
planeId = p.loadURDF("plane.urdf")
```




## Load the robot
startPos is a list of 3 floating-point numbers that represent the initial position of an object in 3D space. The values in the list correspond to the x, y, and z coordinates respectively. In this case, the initial position is set to the origin (0,0,0) which is the point (x,y,z) = (0,0,0) in the 3D space.

startOrientation is a variable that holds the initial orientation of an object in the form of a quaternion. A quaternion is a 4D mathematical object that can be used to represent rotations in 3D space. In this case, startOrientation is set by calling the p.getQuaternionFromEuler function, which takes a list of 3 floating-point numbers representing the Euler angles (in radians) of the object's initial orientation.
The [0,0,0] passed as the argument corresponds to the yaw, pitch and roll of the object in that order.

Together, startPos and startOrientation define the initial position and orientation of an object in the simulation. These values can be used to specify the initial state of an object when it is added to the simulation.


```python
#place the robot at the base position
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])

#load the robot urdf file
boxId = p.loadURDF("/content/pybullet-works/notebooks/meshes/ur5.urdf",startPos, startOrientation)
```




## Adding camera to the scence
The variables pitch, roll, and yaw define the rotation of the camera in 3D space. pitch represents the rotation around the x-axis, roll represents the rotation around the y-axis, and yaw represents the rotation around the z-axis. In this case, the pitch is set to -10 degrees, which will make the camera look slightly downwards, the roll is set to 0, and the yaw is not defined.

upAxisIndex is an integer variable that represents the up axis of the camera. This variable is used to specify which axis of the camera is pointing upwards. In this case, the value is set to 2, which corresponds to the z-axis.

camDistance is a variable that represents the distance of the camera from the target position. In this case, the camera is 1.5 units away from the target position.

pixelWidth and pixelHeight define the resolution of the image captured by the camera. In this case, the image will be 640 pixels wide and 480 pixels tall.

nearPlane and farPlane define the near and far clipping planes of the camera, respectively. Objects closer than the near plane or farther than the far plane will not be visible in the captured image. In this case, the near plane is set to 0.01 units and the far plane is set to 100 units.

fov is the field of view of the camera, measured in degrees. In this case, the field of view is set to 60 degrees.

viewMatrix and projectionMatrix are 4x4 matrices that define the position and configuration of the camera. viewMatrix is computed using the p.computeViewMatrixFromYawPitchRoll function, which takes several arguments such as the target position, distance, yaw, pitch, roll, and up axis index of the camera. projectionMatrix is computed using the p.computeProjectionMatrixFOV function, which takes the field of view, aspect ratio, near and far clipping planes of the camera.

Finally, the p.getCameraImage function is used to capture an image of the simulation. This function takes several arguments such as the width, height, viewMatrix, and projectionMatrix of the camera and returns an image in the form of an array.

```python
%%time
camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 1, 1]
p.setGravity(0, 0, -10)

from google.colab import widgets
import numpy as np
import random
import time
from matplotlib import pylab
grid = widgets.Grid(2, 2)
yaw = 0
for r in range(2):
  for c in range(2):
    yaw += 60
    with grid.output_to(r, c):
      grid.clear_cell()
      pylab.figure(figsize=(10, 5))
      pitch = -10.0
      roll = 0
      upAxisIndex = 2
      camDistance = 1.5
      pixelWidth = 640
      pixelHeight = 480
      nearPlane = 0.01
      farPlane = 100
      fov = 60
      viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                                  roll, upAxisIndex)
      aspect = pixelWidth / pixelHeight
      projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
          
      img_arr = p.getCameraImage(pixelWidth,pixelHeight,viewMatrix,projectionMatrix)
      w = img_arr[0]  #width of the image, in pixels
      h = img_arr[1]  #height of the image, in pixels
      rgb = img_arr[2]  #color data RGB
      dep = img_arr[3]  #depth data
      print("w=",w,"h=",h)
      np_img_arr = np.reshape(rgb, (h, w, 4))
      np_img_arr = np_img_arr * (1. / 255.)
      pylab.imshow(np_img_arr, interpolation='none', animated=True, label="pybullet")
```
<p align="center">
<img src="https://user-images.githubusercontent.com/70435083/215380119-15fcf56b-8aaa-429a-854c-b58bb810fab4.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Robot in all views.</a>
</p>



## Create an animated png



```python

!pip install numpngw
from numpngw import write_apng
from IPython.display import Image


frames=[] #frames to create animated png
for r in range(60):
    yaw += 6
    pitch = -10.0
    roll = 0
    upAxisIndex = 2
    camDistance = 1.5
    pixelWidth = 320
    pixelHeight = 200
    nearPlane = 0.01
    farPlane = 100
    fov = 60
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                                roll, upAxisIndex)
    aspect = pixelWidth / pixelHeight
    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        
    img_arr = p.getCameraImage(pixelWidth,pixelHeight,viewMatrix,projectionMatrix)
    w = img_arr[0]  #width of the image, in pixels
    h = img_arr[1]  #height of the image, in pixels
    rgb = img_arr[2]  #color data RGB
    dep = img_arr[3]  #depth data
    #print("w=",w,"h=",h)
    np_img_arr = np.reshape(rgb, (h, w, 4))
    frame = np_img_arr[:, :, :3]
    frames.append(frame)
print("creating animated png, please about 5 seconds")
%time write_apng("example6.png", frames, delay=100)
%time Image(filename="example6.png")
```


<p align="center">
<img src="https://user-images.githubusercontent.com/70435083/215378974-9468cbe5-50e2-49a3-8a74-4031634cc8f1.gif" alt="ultrasoundgif" class="center">
</p>

<p align="center">
<a href="">UR5 robot interaction with a torid soft body.</a>
</p>

