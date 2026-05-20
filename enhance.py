# open the image 

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms.functional import adjust_gamma
from utils import img2tensor
# from utils import FileClient

path = "/ghome/mpilligua/lowlight/OURdataset/nikon/test/high/Test1-F_90-SS_200-x_3127-y_3290-B_254.png"
img_o = cv2.imread(path)
img_o_tensor = img2tensor(img_o, bgr2rgb=True, float32=False)

img = adjust_gamma(img_o_tensor, 1/3)

# do histogram equalization
img = img.permute(1, 2, 0)
img = img.numpy()

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(img_o[:,:,::-1])
ax[1].imshow(img)
plt.savefig("enhanced.png")

