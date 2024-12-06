---
layout: post
title:  SignLangauge - A Review of entire system.
date:   2023-08-22 16:40:16
description: Sign Language Video Generation.
tags: Projects
categories: SignLangauge
---

# Introduction to Sign Language

Signers communicate with peers using hand signs. Thus we are require of systems that can help us to convert these signs to words and inverse as well.

There are 3 different things in sign langauge world.
1. Sign Language Recognition - Convert signs to words.
2. Sign Language Generation - Convert words to signs.
3. Sign Language Translation - Convert signs to words and words to signs.

All the datasets in the sign langauge would have a video + gloss + text. Based on the dataset availability we procced for the type of the task. A lot of effort is put in the sign language recognition. I was more of a CV engineer, I did like the generation part. My goal is to make a system where an user enters the text input and the system generates the video of the sign language. The output video is meant to be a human realistic sign language video.

The things I am intrested in to look for are:
1. Text to Pose
3. Text to Video

Anyways in both the procedure the common ground is to have text input and a video output. When I say text, the input would be a spoken langauge like english, hindi, french...etc. The output would be a video of the sign language( skeleton based one or a human realistic one).

## Research Papers

### 2023
1. Ham2Pose: Animating Sign Language Notation into Pose Sequences.[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Arkushin_Ham2Pose_Animating_Sign_Language_Notation_Into_Pose_Sequences_CVPR_2023_paper.html)
2. Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation
3. Co-speech Gesture Synthesis by Reinforcement Learning with Contrastive Pre-trained Rewards.

### 2022
1. BEAT: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis
2. Audio-Driven Stylized Gesture Generation with Flow-Based Model.
3. Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic Sign Language Production.
4. Audio-Driven Neural Gesture Reenactment With Video Motion Graphs.
5. Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation.
6. SEEG: Semantic Energized Co-Speech Gesture Generation.
7. Low-Resource Adaptation for Personalized Co-Speech Gesture Generation.
8. Learning Unseen Emotions from Gestures via Semantically-Conditioned Zero-Shot Perception with Adversarial Autoencoders
9. Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks.

### 2021    
1. Speech Drives Templates: Co-Speech Gesture Synthesis With Learned Templates.
2. Audio2Gestures: Generating Diverse Gestures From Speech Audio With Conditional Variational Autoencoders.
3. Mixed SIGNals: Sign Language Production via a Mixture of Motion Primitives.
4. Towards Fast and High-Quality Sign Language Production.
5. Speech2AffectiveGestures: Synthesizing Co-Speech Gestures with Generative Adversarial Affective Expression Learning.
6. Progressive Transformers for End-to-End Sign Language Production.
7. Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional-Mixture Approach.
8. Neural Sign Language Synthesis: Words Are Our Glosses.

### 2018
1. Sign Language Production using Neural Machine Translation and Generative Adversarial Networks.