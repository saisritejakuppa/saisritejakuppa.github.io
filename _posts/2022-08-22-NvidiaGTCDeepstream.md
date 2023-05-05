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

----

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
