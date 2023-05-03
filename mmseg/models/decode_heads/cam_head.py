import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class CAMHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        self.loss_weight = kwargs.pop('loss_weight', 1.0)
        super(CAMHead, self).__init__(**kwargs)
        del self.conv_seg
        self.conv_cam = nn.Conv2d(
            self.in_channels,
            self.num_classes,
            kernel_size=1,
            bias=False
        )
        self.loss_decode = nn.BCEWithLogitsLoss()
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        cam_map = self.conv_cam(x)
        return self.relu(cam_map), F.adaptive_avg_pool2d(cam_map, 1)
    
    def forward_train(self, 
                      inputs, 
                      img_metas, 
                      gt_semantic_seg, 
                      seg_weight=None):
        cam_map, scores = self.forward(inputs)
        losses = self.losses(scores, gt_semantic_seg, seg_weight)
        return losses, {'cam_map': cam_map, 'cam_scores': torch.sigmoid(scores)}
    
    @force_fp32(apply_to=('cam_map', ))
    def losses(self, cls_scores, seg_label, seg_weight=None):
        loss = dict()
        cls_label = self._get_cls_label(cls_scores, seg_label)


        loss['loss_cam'] = self.loss_decode(cls_scores, cls_label) * self.loss_weight
        return loss
    
    def _get_cls_label(self, cls_scores, seg_label, ignore_index=255):
        batch_size = seg_label.shape[0]

        cls_label = torch.zeros_like(cls_scores)
        for i in range(batch_size):
            tmp = seg_label[i].unique(sorted=True)
            if tmp[-1] == ignore_index:
                tmp = tmp[:-1]
            cls_label[i, tmp] = 1
        return cls_label
    
    def init_ema_weights(self, aux_head):
        for p in self.parameters():
            p.detach_()
        
        self.conv_cam.training = False
        
        for ema_p, p in zip(self.parameters(), aux_head.parameters()):
            if not ema_p.data.shape:
                ema_p.data = p.data.clone()
            else:
                ema_p.data[:] = p.data[:].clone()

    def update_ema_weights(self, aux_head, alpha):
        for ema_p, p in zip(self.parameters(), aux_head.parameters()):
            if not ema_p.data.shape:
                ema_p.data = \
                    alpha * ema_p.data + (1 - alpha) * p.data.clone()
            else:
                ema_p.data[:] = \
                    alpha * ema_p.data[:] + (1 - alpha) * p.data[:].clone()
        

        

