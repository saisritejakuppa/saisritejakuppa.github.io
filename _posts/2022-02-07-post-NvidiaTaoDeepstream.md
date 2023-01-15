---
title: "Installing Nvidia Tao and DeepStream"
date: 2022-07-02T15:34:30-04:00
categories:
  - blog
tags:
  - Computer Vision 
  - DeepStream
  - Nvidia Tao
---


# Nvidia Tao Tutorial


# Installing Docker 

```python
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

 sudo mkdir -p /etc/apt/keyrings
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

 echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null


  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```


My system specs:

```python
$user:nvidia-smi
Sat Jul  2 17:13:57 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  On   | 00000000:06:00.0 Off |                    0 |
| N/A   49C    P0    59W / 250W |      0MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```




# Installing DeepStream for Python

The following sequence of steps can install the deepstream in docker in a proper format.

```
docker pull nvcr.io/nvidia/deepstream:6.1-samples
```


```
apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev
```



```
cd /sources
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
git submodule update --init
```


```
apt-get install -y apt-transport-https ca-certificates -y
update-ca-certificates

```



```
cd 3rdparty/gst-python/
./autogen.sh
make
sudo make install
```


```
cd deepstream_python_apps/bindings
mkdir build
cd build
cmake ..
make
```


To test run the a file using deepstream.

```
cd deepstream_python_apps/apps/deepstream-test2/
python3 deepstream_test_2.py test.h264 1
```
