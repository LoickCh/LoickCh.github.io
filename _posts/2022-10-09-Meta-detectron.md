---
layout: post
title: Detectron & detectron2
date:   2022-10-09
description: Popular Python Meta Research github repository.
tags: python
categories: library
---

# Detectron and Detectron 2

## I. Introduction

Detectron and <a href=" object detection, segmentation and other visual recognition tasks." >
detectron2 </a> are the most starred Python github repository from 
Meta Research with respectively more than 25k and 22k stars. Both aim to implement
and maintain state-of-the-art algorithms for multiple visual recognition tasks.
The first version uses Caffe2 deep learning framework while the the second one uses Pytorch. 

Among possible tasks, we can perform pose estimation, object classification and 
(semantic, instance and panoptic) segmentation. 

## II. Usecase

To illustrate the library, we can imagine we are ornithologists who need to 
classify a lot of bird images. As birds share many 
common attributes, we would like to crop our database on birds in order to 
classify their species in a second step.  

Let us take the Caltech-UCSD Birds-200-2011 database containing many bird images
taken from Flickr. It contains 200 bird species for 11,788 images.

After installing the library using pip, we import usefull functions, libraries and
modules:

```python
# Install detectron2 in a notebook
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import glob 
import numpy as np
import torch, cv2
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

Then we need to select a pre-trained model among those available <a href="https://github.com/facebookresearch/detectron2/blob/main/detectron2/model_zoo/model_zoo.py"> here </a>. To compare models, we can look at the
<a href="https://github.com/airsplay/py-bottom-up-attention/blob/master/MODEL_ZOO.md"> benchmark </a>.
In our case, we use a Faster-RCNN like pre-trained model. In addition, we specify a 
confidence threshold. Only results with a confidence higher than the threshold will
be outputed. 

```python 
TASK="COCO-Detection"
MODEL="faster_rcnn_X_101_32x8d_FPN_3x"

cfg = get_cfg() # Get a copy of the default config.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # confidence threshold.
cfg.merge_from_file(model_zoo.get_config_file(f"{TASK}/{MODEL}.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{TASK}/{MODEL}.yaml")
```

Then we select the default predictor and we feed it a single image. In background, it loads weights
using the given configuration and performs several preprocessing steps automatically.

```python
# Create a simple end-to-end predictor with the given config that runs on single 
# device for a single input image.
predictor = DefaultPredictor(cfg) 

# Prediction
files=glob.glob("data/*.jpg")
im = cv2.imread(files[0])
outputs = predictor(im)
```

DefaultPredictor's call outputs a dictionary containing metadata about images and
predictions depending on the task and the model used. In our case, it predicts
 bounding boxes and class. By default bounding boxes are contained in a Box object 
having the following format: $$x_{min}, y_{min}, x_{max}, y_{max}$$.

```python
box=outputs['instances'].pred_boxes.tensor.cpu().numpy()[0] # Get outputs
xmin,ymin,xmax,ymax = np.floor(box).astype('int') # Unpack bounding box
cropped_im=im[ymin:ymax, xmin:xmax] # Crop
```

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Blog/2022-10-09/Detectron_crop.png" title="Decoder architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Original image and its cropped (and resized) version.
</div>

Personally, I used detectron2 during a bird-classification challenge to pre-process the whole 
database. I found it very easy to use using the official starting <a href="https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5"> notebook </a>. My cropping was a bit more complex, 
I used three different predictors, I did a square crop with a pixel tolerance 
at the border but the idea is still the same. 

