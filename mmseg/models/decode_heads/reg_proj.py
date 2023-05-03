import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class RegProjHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(RegProjHead, self).__init__(**kwargs)
        del self.conv_seg
        self.reg_proj = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=False
        )
        self.loss_decode = nn.MSELoss()
    
    def forward(self, inputs):
        reg_proj = self.reg_proj(inputs)
        return reg_proj