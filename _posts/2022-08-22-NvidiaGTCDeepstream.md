---
layout: post
title:  Solving Grand Challenges in Video Analytics
date:   2022-08-22 16:40:16
description: Embeddings
tags: DeepStream
categories: DeepStream
---


Alright now you know how to train a deep learning model, what's next. You wanna deploy it and check it how its performing. If its performing bad, how to proceed further. If its performing good, how to scale it.

In this blog we are gonna deal with all these stuff about deployment and model retraining for a high accuracy.





<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Place where you wanted to deploy your model usally.
</div>



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 2.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 3.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Deploying the model in Deepstream.
</div>


----

## Deepstream

The RTSP is a steam of video frames that are given to the video management software and AI models for inferences. Once the inference is done on the image, the database is stored and the inferences are pushed to custom analytical dashboard. Usually we have mutliple source of cameras and we build a pipeline for each of them. This can work for a small scale but when we have a large scale of cameras, this is not a good approach. We need better apporaches to handle this. If we can combine output from multiple cameras we can even get better results for applications like traffic management, crowd management etc. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Current Methods.
</div>





## Features in Upgraded Deepstream

Now we have a warehouse where a lot of people keep moving in it. One of the core application is to track and detect the person all the time. 

A detector module helps to draw a bouding boxes around a person and the tracker traces the path of the person. We find these modules often in any system. The tracker is used to check if a person enters an unauthorized area.

Now image a person moves from one room to another, since there is a switch of a person in cam 1 to 2, a new tracker is allocated to same person, which is not what we wanted. In such situations we need to have a multi camera tracking system. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 4.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Deploying the model in Deepstream.
</div>


## Challenges in Multiple Tracking System

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 6.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Challenges in Multiple Tracking System.
</div>

## Generating CG Dataset and Training( Omniverse )

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 15.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CG dataset Creation + Model Training.
</div>


## Bulding the CG Dataset

One the challenges in training a model is to get a good datasets. We usually do have the real datasets which is blurry. The other problem is, when it comes to things like tracking the ground truth is not available. Annotating such tasks is very difficult. These are the reasons why we need to approach for CG dataset. Omniverse gives you a set of predefined enviroments to deal with. We can use these environments to generate the dataset. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 9.png" class="img-fluid rounded z-depth-1" %}
    </div>
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 14.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Nvidia Omniverse, place to built virtual env.
</div>


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 10.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 11.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Building a digital twin of the warehouse.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 12.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Building a Virtual Humans in warehouse.
</div>



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 13.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Output of CCTV photages generated from CG environment.
</div>





## Training the neural nets.

In further training, along with the real data we can use the CG dataset for training the neural network. This will help the model to learn the features of the object in a better way. Also there will be any mistakes in the dataset given by the CG.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 8.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    New training pipeline.
</div>



RE - ID is a process where the model is used to detect the same person in multiple cameras. This is done by using the features of the person. The features are extracted from the person and stored in a database. When a person is detected in a new camera, the features are extracted and compared with the database. If the features are matched, then the person is the same.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 16.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 17.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Re-ID for person tracking.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 21.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Performance on multiple machines.
</div>



## Building Multicamera perception system

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 7.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Multiple camera perception system.
</div>



We need to upgrade the system from single camera to multiple camera. This is done by using the multiple camera perception system. This system is used to detect the objects in multiple cameras. 


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 18.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Multiple camera modules. 
</div>


The multi tracking system have:
1. Pixel to Physical mapping: This is used to map the pixel to the physical location of the object. This is done by using the camera calibration.

2. Behavioural analysis: This is used to analyse the behaviour of the object. This is done by using the object detection and tracking.

3. Matching Process: This is used to match the object in multiple cameras. This is done by using the Re-ID.



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 19.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Multiple camera modules. 
</div>



# Microservices( Perception, Data storage, Dashboard)


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 22.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Perception module.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 23.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Behaviour module.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 24.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Storage module.
</div>



## Metrapolis 

This is a platform where you can build all the things from end to end. Yet, omniverse is so expensive and deepstream doesnt have a great document. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 26.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Solving_Grand_Challenge/Untitled 27.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Metrapolis.
</div>



