---
title: "Sign Langauge Part 2: Generating the Database"
date: 2022-07-02T15:34:30-04:00
categories:
  - blog
tags:
  - Computer Vision 
  - Sign Langauge
  - NLP
---

This blog post is about generating the database from the videos to train the GAN. I am using the German dataset mentioned in the previous blog for this task. The dataset is available in the following link: [meinedgs](https://www.sign-lang.uni-hamburg.de/meinedgs/overview/start.html). I am taking a few sample vidoes and trying to make things on top of that. The dataset is having 2 types of videos, one is the video of the signer signing the word, and the other is the signer signing the entire sentence. I am using the entire sentence videos for this task. The dataset is having the following files:

I am using google colab to generate the database. The following code is used to download the videos from the website.


```python
!git clone https://github.com/BenSaunders27/meineDGS-Translation-Protocols.git
import os
from tqdm import tqdm
import pandas as pd
data = pd.read_csv('/content/meineDGS-Translation-Protocols/mDGS-V/mDGS_Protocol_Train.csv', sep = '|')


from tqdm import tqdm

# data['gloss']
def MakeIndividualWords(dataframe):
  words_info = dict()
  for no,d in tqdm(enumerate(dataframe.tolist())):
    try:
      words = d.split(' ') 
      for word in words:
        word_specific, startframe, endframe = word.split('/')
        if word_specific not in words_info.keys():
          words_info[word_specific] = no
    except:
      pass
  return words_info


from pathlib import Path
import os

from pathlib import Path
import os

class PrepareData:
  def __init__(self, dataframe, 
               download_vids = False,
               downloadpath_vids = '/content/dataset/vids',

               download_poses = False,
               downloadpath_poses = '/content/dataset/poses',


               makedataset =  True,
               makedatasetpath = '/content/signvideos/'):
    


    super().__init__()
    self.df = dataframe
    self.downloadpath_vids = downloadpath_vids
    self.downlaodpath_poses = downloadpath_poses

    self.MakeURL()
    self.GetVidDataset()


    if download_vids == True:
      self.DownloadVids()

    if download_poses == True:
      self.DownloadOpenPose()

    if makedataset == True:
      self.MakeDataset()

  def MakeURL(self):
    self.fnames = self.df['filename'].tolist()
    self.start = self.df['start_time'].tolist()
    self.end   = self.df['stop_time'].tolist()
    self.gloss = self.df['gloss'].tolist()
    self.individualwords = MakeIndividualWords(self.df['gloss'])
    

  def GetVidDataset(self):
    print(self.fnames)
    self.vids_name = set([i.split('-')[0][:-1] for i in self.fnames])

  def DownloadVids(self):
    if not os.path.exists(self.downloadpath_vids):
      os.makedirs(self.downloadpath_vids)
    for vid in tqdm(self.vids_name):
      vidname = f'https://www.sign-lang.uni-hamburg.de/meinedgs/release2/videos/{vid}/{vid}.mp4'
      !wget $vidname -P $self.downloadpath_vids
      # break

  
  def DownloadOpenPose(self):
    if not os.path.exists(self.downlaodpath_poses):
      os.makedirs(self.downlaodpath_poses)
    for vid in tqdm(self.vids_name):
      vidname = f'https://www.sign-lang.uni-hamburg.de/meinedgs/openpose/{vid}_openpose.json.gz'
      !wget $vidname -P $self.downlaodpath_poses


  def MakeDataset(self, fps = 50.0,
                  makedatasetpath = '/content/signvideos/',
                  cropped_imgs_path = '/content/cropped_vids',
                  downloadpath = '/content/drive/MyDrive/Teja_SignLangauge/GovernmentProject'):
    
    os.makedirs(cropped_imgs_path)

    for word,rowno in tqdm(self.individualwords.items()):
      Path(os.path.join(makedatasetpath, word)).mkdir(parents=True, exist_ok=True)

      rowinformation = self.df.iloc[rowno]
      filename,	camera	,ger_text	,gloss	,start_time	,stop_time = rowinformation.values
      # print(filename,	camera	,ger_text	,gloss	,start_time	,stop_time)

      ger_text= ger_text.replace(' ', '-')       #replace the spaces
      sh,sm,ss,sms = convertmins(start_time	, 50)
      eh,em,es,ems = convertmins(stop_time, 50)

      vidname_org = filename.split('-')[0][:-1] + '.mp4' 
      vidname = os.path.join(downloadpath, vidname_org)


      # if os.path.exists(vidname):
      #     print("File exists")


      comands_crop = f'ffmpeg -i {vidname} -ss {sh:02d}:{sm:02d}:{ss:02d}.{sms:03d} -t {eh:02d}:{em:02d}:{es:02d}.{ems:03d} -c copy {cropped_imgs_path}/{ger_text}_{vidname_org}'
      # print(comands_crop)
      os.system(comands_crop)


      # break

 
    
downloadpath = '/content/drive/MyDrive/Teja_SignLangauge/GovernmentProject'
# P1 = PrepareData(data, downloadpath_poses = downloadpath, download_poses = True)
P1 = PrepareData(data, downloadpath_poses = downloadpath, makedataset = True)
```



