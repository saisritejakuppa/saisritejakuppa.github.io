---
layout: post
title:  Deep Learning Diffusion  - Math and Code
date:   2022-08-22 16:40:16
description: Introduction to Diffusion in Deep Learning. We will be generating unconditional images.
tags: Deep-Learning
categories: Deep-Learning
---

## Diffusion vs GAN

For  image generation, we have been using GANs for a considerable time. However, training GANs can be challenging due to the need to control multiple loss functions and assign individual weights to them. To overcome this difficulty, we turn to diffusion models.

Diffusion models have shown promising results in generating images with high fidelity and diversity, with a lower incidence of model collapse compared to GANs. Inspired by the impressive outputs of midjourney and DALL-E, I decided to build my own custom application using diffusion models.

If you are new to this, and do not want to delve into the mathematical details, [diffusers](https://huggingface.co/docs/diffusers) can be a useful starting point. They offer pipelines and pre-built components to experiment with. However, as a researcher who frequently reads papers, I find it important to understand the underlying math to achieve the best results.


## The Diffusion Models

Lets begin the journey.
I am assuming you have some good grasp on deep learning tearms and moving with that in mind. 

<em>Model Building</em> and <em>Training Process</em>, these are the only things you need to focus on if have already trained a simple CNN in pytorch. Lets start with each of them.


## Model Building

In the deep learning world, signals can be of format audio, vision(image in 2d, 3d), text. These are my main focus. 

> Any information is converted to tensor, these tensors from higher dimension brought down to lower dimension, these are called <em>Embeddings</em>. Its all about embeddings in the end of the day.

I like images, so in today experiment we will start with generating new images without any condition. If you like text/ audio, find a way to convert it to tensors and try the experiments from the research as well.


### Core Blocks of Diffusion Model

The core blocks of the diffusion model:
1. Downscaling Blocks
2. Upscaling Blocks
3. Residual Blocks
4. Convolutional Blocks
5. Attention Blocks


Lets take an image 512 X 512 X 3, the number of parameters are 786432, doing computations on such high number is not possible unless we deal with lower dimensions. So, we need to downscale the image to lower dimensions. This is where the <em>downscaling blocks</em> come into the picture.

Now what we do is take the lower dimension image( latent image ), add information to it and remove information from it. This is where the <em>residual blocks</em>, <em>convolutional</em> and <em>attention blocks</em> come into the picture.


Convolutions picks the information from the neighbouring pixels and adds it to the current pixel. So it looks around a few neighbouring pixels.
In attension, each pixel looks at all the other pixels in the entire feature map. So, it is a global operation. Thus the reason transformers are better than the convs. 
Residual connections helps during the backwards propagation for the flow of the gradients.


The refined lower dimension image does consist of the information of the higher dimension image. So, we need to upscale the image to the original dimension. This is where the <em>upscaling blocks</em> come into the picture.




-----------


### Convolutional Block

Convolution blocks helps to retain the information of the neighbouring pixels. This helps to presever from low freq to high freq information. 

```python

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
```



-----------

### Downscaling Blocks

Downscaling block helps to reduce the dimension of the image. This is done by maxpooling and convolution. The maxpooling helps to reduce the dimension of the image. The convolution helps to retain the information of the neighbouring pixels.

Take a focus on the variable t, but t is an embedding or extra information we are giving to the model. This can be any vector.


```python

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
```


-----------



### Upscaling Block
Upscaling block you can think of it as the reverse of the downscaling block. It helps to upscale the image to the original dimension.

we have discussed about the resnet block, we didnt rather implement a whole block, we embedded into the code as a line, where we take information from the downscale and add the information to the upscale blocks. This helps to preserve the information of the higher dimensions of the feature maps and the backpropagation is easy as well.

Take a focus on the variable t, but t is an embedding or extra information we are giving to the model. This can be any vector.

```python

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
```

-----------


### Self Attension Block

Self attention picks only important features and helps you to retain the the block things which are essential. 
There is self attention and cross attention. Self Attension is used to get important features from the images. Imagine now you want to cluster up two images, then you start looking for cross attenstion. The other one need not be an image alone, also text/audio. Tensors again. 

```python

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        
        # embed_dims, num_heads, 
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        
        self.ln = nn.LayerNorm([channels])
        
        #convert to req final shape
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)   #[1,4,16,16] -> [1,256,4]
        x_ln = self.ln(x)                                                     #[1,4,16,16] -> [1,256,4]               
        
        #query, key, value -> attn_output, attn_output_weights
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)                       #[1,256,4] -> [1,256,4]
                
        #resnet connection
        attention_value = attention_value + x
        
        #add the feed forward layer to get more features and attenuate the noise
        attention_value = self.ff_self(attention_value) + attention_value           #[1,256,4] -> [1,256,4]
        
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)  #[1,256,4] -> [1,4,16,16]
```

-----------


### Unet Model
Unet is a model introduced intially for bio medical segmentation, but soon it spread in every domain. The unet consist of downscaling + bottleneck + upscaling blocks. 

There are skip connections from downsacling to upscaling blocks to preserve the information from the original image.

```python

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        #downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        #bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)


        #upscaling
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
```





<strong>Positions Encoding block</strong>

The position encoding block helps as gudiance of the attension block to tell which pixels are close and which pixels are far for a certain pixel in the image. We use sine and cosine because of the reapeating frequency over a certain range. This position embeddings tells the location of the pixels, so that the attention block knows how can they be related to each other.

```python
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
```


