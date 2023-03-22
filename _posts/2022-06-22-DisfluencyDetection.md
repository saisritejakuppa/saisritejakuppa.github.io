---
layout: post
title:  Disfluency detection using deep learning.
date:   2022-09-22 16:40:16
description: A blog about disfluency detection using deep learning.
tags: SpeechProcessing
categories: SpeechProcessing
---

# DisFluency Detection Using Deep Learning

Disfluency detection is the task of identifying and classifying disfluent speech segments in spoken language. It can be useful in many applications such as speech therapy, speech recognition and natural language processing. Deep learning methods have been used to detect disfluencies with high accuracy. These methods use neural networks, such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), to analyze the acoustic and/or linguistic features of the speech signal. The neural network is trained on a dataset of labeled speech segments and then can be used to classify new speech segments as fluent or disfluent. The use of deep learning methods for disfluency detection has shown promising results and has the potential to improve the performance of disfluency detection systems.


Softwares used:
1.  Python 
2.  Praat
3.  TensorFlow

PRATT is a software tool that is used for the purpose of speech processing. It is designed to help individuals with speech impairments, such as dysarthria, to improve their speech intelligibility. The software uses advanced speech processing algorithms to analyze the user's speech and provide real-time feedback on various aspects of speech, such as pitch, loudness, and duration. Users can also practice specific speech exercises and track their progress over time. PRATT can be used in a clinical setting by speech therapists or can be used as a self-help tool by individuals with speech impairments. The software is typically used in conjunction with other speech therapy techniques and interventions. Overall, PRATT is a powerful tool that can help individuals with speech impairments to improve their speech intelligibility and communication skills.


# Complete Algorithm 

<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163709544-7e40bb40-1b57-42ee-90c9-70397f79e71e.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Complete Algorithm.</a>
</p>

## Process of the Project
1.  Build a website and connect it to the AWS to save the recorded data which we get live.
2.  Data Analysis and Data Cleaning.
3.  Data Agumentations and Speech Processing tricks for a better output.
4.  Build a Deep Learning Model to Predict the disfluency in speech.
5.  Praat Software to get meta data.


# 1. Website Building and connecting to Github
We have build a simple website using bootstrap and HTML, basic javascript to do some operations in the website. Flask Package is used to deploy the website in Heroku. We have written script in such way that the audio collected is saved in AWS S3 bucket. Then we get to download the data from the bucket for further process. The website consits of  a GUI to record speech data, the person is given a set of questions to explain, and is about speak about 3 mins regarding ques. This is recorded and stored. 
The link of the website is attached below. 


https://myprosody.herokuapp.com/

<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163707254-5e810fcd-d281-41db-a81a-6bb1b35e72f7.png" alt="ultrasoundgif" class="center">
<br>
<a href="">Webpage for data collection.</a>
</p>


# 2. Data Analysis Part

1.  The data we get is recorded from various devices and different browser extensions. So sampling rate is set accordingly for proper pre processing.
2.  Human Pitch for Men(100- 120Hz) and Women(300Hz), so low frequency information is necessary and high frequency is discarded. To do this we used a low pass filter.
3.  A window is 10 seconds is sampled so that we can send in limited features to detect disfluencies.
4.  The Mel scale mimics how the human ear works, with research showing humans don't perceive frequencies on a linear scale. Humans are better at detecting differences at lower frequencies than at higher frequencies. So we have used mel spectrograms as an input.




Librosa is a python library for analyzing and manipulating audio files. It is designed to be easy to use, and provides a wide range of functionality for tasks such as feature extraction, audio segmentation, and audio visualization. Some of the key features of librosa include:
-Support for a wide range of audio file formats and codecs
-Tools for loading and resampling audio data
-Tools for audio feature extraction, including Mel-frequency cepstral coefficients (MFCCs), chroma, and tempo estimation
-Tools for audio segmentation, such as beat tracking, onset detection, and pitch estimation
-Tools for audio visualization, including waveform plots, spectrograms, and tonnetz plots
-Integration with other popular python libraries such as numpy, scipy and matplotlib.

Librosa is widely used in the field of music information retrieval and audio processing, and is a popular choice for researchers and developers working in these areas. It is open source and actively maintained.

I have used the Librosa library to extract the spectrograms and do the augumentations.
<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163707377-24d26e11-ce0a-4934-90fb-4d64911ea4af.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Data Collection Workflow.</a>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163709124-d00760eb-f70a-4c74-ab29-eb55882fdb7c.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Generating Spectrogram.</a>
</p>




# 3. Data Agumentations
Data augmentation is a technique used to artificially increase the size of a dataset in order to improve the performance of machine learning models. One common data augmentation technique for audio data is to black out pixels in the spectrograms. Spectrograms are visual representations of audio data, where the x-axis represents time, the y-axis represents frequency, and the intensity of each pixel represents the amplitude of the audio signal at that point in time and frequency. By randomly blacking out pixels in the spectrograms, the model is forced to learn to be robust to missing information, which can improve its ability to generalize to new unseen data.

This technique can be implemented by applying a mask to the spectrogram, where certain pixels are randomly set to zero. The mask can be applied with different probability, and can be applied in a localized way such as blacking out a random square region of the spectrogram. This can be done using python library such as numpy, scipy and librosa.

Blacking out pixels in the spectrograms can be especially effective when combined with other data augmentation techniques, such as time shifting, pitch shifting, and adding noise to the audio signal. By applying these different types of data augmentation, the model can learn to be robust to a wide range of variations in the input data, which can lead to improved performance.

<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163709280-9a2191fa-a436-44d8-b980-4f325bef81cf.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Data Augumentations.</a>
</p>



# 4. Deep Learning Architecture
The model consist of resnet blocks to observe patterns in spectrograms and predict output. The output can predict both the labels, since we have used multilabel classification using sigmoid at the end for each classifier.

The CNN architecture is designed to be able to learn patterns in the spectrograms, which can then be used to make predictions about the audio data. The specific architecture used in this model is a variant of the ResNet blocks.

ResNet blocks are a type of building block used in CNNs that are designed to be able to learn residual representations of the input data. This means that the model is able to learn the differences between the input data and a set of learned features, rather than trying to learn the features themselves. This can make the model more robust and able to generalize better to new unseen data.

In this model, the ResNet blocks are used to observe patterns in the spectrograms, and the output of the model is used to make predictions about the audio data. The model is trained using a multilabel classification task, which means that it is able to predict multiple labels for a given input. The final output layer of the model uses a sigmoid activation function for each classifier, which allows the model to output a probability for each label.

The use of ResNet blocks and a multilabel classification task allows the model to make predictions about the audio data with high accuracy. Additionally, the use of a sigmoid activation function in the final output layer allows the model to output probabilities for each label, which can be useful for making decisions based on the model's predictions.
<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163709308-5bcf16f9-f2b0-4eef-bee3-2151a9f492a4.png" alt="ultrasoundgif" class="center">
<br>
<a href="">Model Architecture.</a>
</p>


# 5. Additional Data
Certain audio clips contain no information and noise, for such data using a deep learning model is a waste of computation, so we use praat software to detect such things and help it to predict as long pauses. The algorithm is good enough to predict all the filler long pauses and is highly accurate. Hence it can deal in all the situations.


# Results
<p align="center">
<img src="https://user-images.githubusercontent.com/48018142/163709282-f0ddeffc-2933-4d6b-83a3-c977bcf5e93e.JPG" alt="ultrasoundgif" class="center">
<br>
<a href="">Results.</a>
</p>