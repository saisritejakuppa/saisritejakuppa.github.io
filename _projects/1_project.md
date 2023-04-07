---
layout: page
title: DisFluency Detection Using Deep Learning
description: Identification of disfluent words in a speech signal
img: assets/img/projects/disfluency.jpeg
importance: 1
category: work
---

Softwares used:
1.  Python 
2.  Praat
3.  TensorFlow

# Complete Algorithm 
![image](https://user-images.githubusercontent.com/48018142/163709544-7e40bb40-1b57-42ee-90c9-70397f79e71e.JPG)

## Process of the Project

1.  Build a website and connect it to the AWS to save the recorded data which we get live.
2.  Data Analysis and Data Cleaning.
3.  Data Agumentations and Speech Processing tricks for a better output.
4.  Build a Deep Learning Model to Predict the disfluency in speech.
5.  Praat Software to get meta data.



# 1. Website Building and connecting to Github
We have build a simple website using bootstrap and HTML, basic javascript to do some operations in the website. Flask Package is used to deploy the website in Heroku. We have written script in such way that the audio collected is saved in AWS S3 bucket. Then we get to download the data from the bucket for further process. The website consits of  a GUI to record speech data, the person is given a set of questions to explain, and is about speak about 3 mins regarding ques. This is recorded and stored. 
The link of the website is attached below. 



<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://user-images.githubusercontent.com/48018142/163707254-5e810fcd-d281-41db-a81a-6bb1b35e72f7.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Heroku App.
</div>

# 2. Data Analysis Part

1.  The data we get is recorded from various devices and different browser extensions. So sampling rate is set accordingly for proper pre processing.
2.  Human Pitch for Men(100- 120Hz) and Women(300Hz), so low frequency information is necessary and high frequency is discarded. To do this we used a low pass filter.
3.  A window is 10 seconds is sampled so that we can send in limited features to detect disfluencies.
4.  The Mel scale mimics how the human ear works, with research showing humans don't perceive frequencies on a linear scale. Humans are better at detecting differences at lower frequencies than at higher frequencies. So we have used mel spectrograms as an input.

![image](https://user-images.githubusercontent.com/48018142/163707377-24d26e11-ce0a-4934-90fb-4d64911ea4af.JPG)
![image](https://user-images.githubusercontent.com/48018142/163709124-d00760eb-f70a-4c74-ab29-eb55882fdb7c.JPG)
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://user-images.githubusercontent.com/48018142/163707377-24d26e11-ce0a-4934-90fb-4d64911ea4af.JPG" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://user-images.githubusercontent.com/48018142/163709124-d00760eb-f70a-4c74-ab29-eb55882fdb7c.JPG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Work Flow of the project.
</div>




# 3. Data Agumentations
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://user-images.githubusercontent.com/48018142/163709280-9a2191fa-a436-44d8-b980-4f325bef81cf.JPG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Data Augmentation.
</div>

# 4. Deep Learning Architecture
The model consist of resnet blocks to observe patterns in spectrograms and predict output. The output can predict both the labels, since we have used multilabel classification using sigmoid at the end for each classifier.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://user-images.githubusercontent.com/48018142/163709308-5bcf16f9-f2b0-4eef-bee3-2151a9f492a4.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Deep Learning Architecture.
</div>

# 5. Additional Data
Certain audio clips contain no information and noise, for such data using a deep learning model is a waste of computation, so we use praat software to detect such things and help it to predict as long pauses. The algorithm is good enough to predict all the filler long pauses and is highly accurate.


# Results
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://user-images.githubusercontent.com/48018142/163709282-f0ddeffc-2933-4d6b-83a3-c977bcf5e93e.JPG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results of the project.
</div>
