import os

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from numpy import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class Blend(object):
    def __init__(self, c=[], root='', n_class=19):
        self.c = c
        self.root = root
        self.n_class = n_class
    
    def __call__(self, results):
        img = results['img']
        lbl = results['gt_semantic_seg']
        img_name = results['filename'].split('/')[-1]
        
        if img.shape[:2] != lbl.shape:
            return results

        img_s2t = Image.open(os.path.join(self.root, img_name))
        img_s2t = transforms.Resize(
            size=(img.shape[0], img.shape[1]))(img_s2t)
        img_s2t = transforms.ToTensor()(img_s2t)
        img_s2t = np.array(
            img_s2t.permute(1, 2, 0) * 255, dtype=np.uint8)
        
        mask = np.zeros_like(lbl)
        for i in self.c:
            mask += (lbl == i)
        mask = mask > 0
        img[mask] = img_s2t[mask]

        results['img'] = img

        return results