import numpy as np
import cv2
import torchvision.utils as tvu


def normalize(img):
    return (img + 1) / 2.0

# imgs is an array of torch tensors
def fuse(imgs):
    imgs = [np.array(e) for e in imgs]
    imgs = [normalize(img[0].transpose(1, 2, 0))*255 for img in imgs]
    res = cv2.hconcat(imgs)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return res
