---
layout: post
title:  SignLangauge - A Review of entire system.
date:   2022-09-22 16:40:16
description: A detailed look at the end to end system for sign langauge video generation.
tags: SignLangauge DeepLearning
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


