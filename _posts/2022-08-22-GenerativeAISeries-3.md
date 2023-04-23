---
layout: post
title:  Deep Learning Series - Part 3
date:   2022-08-22 16:40:16
description: Variational Auto Encoders - VAE
tags: Deep-Learning GenerativeAI
categories: VAE
---

Variatational auto encoder consists of encoder and decoder. Encoder is used to encode the input image to a latent space and decoder is used to decode the images from latent space.

Building a VAE is the one of the trickest part in deep learning. We will be using the same tools and techniques we learned in deep learning series to build a VAE.

There are so many variants of VAE's. We will be using the most basic one to build a VAE and go to a complex architectures.

The latents are high level embeddings of the images. If the embeddings are smooth enough we can generate new images from the embeddings, while traversing on it.

