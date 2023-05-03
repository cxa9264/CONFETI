import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import force_fp32
from torch.distributions.multivariate_normal import MultivariateNormal

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..losses.utils import weight_reduce_loss

@HEADS.register_module()
class ContrastiveHead(BaseDecodeHead):

    def __init__(self, proto_dims=512, top_k=128, num_samples=128,
                 momentum=0.01, tau=0.1, ema=False, hard_sample=True, loss_intra=True, thresh=0.6, proto_type='centroid', **kwargs):
        self.loss_weight = kwargs.pop('loss_weight', 1.0)
        self.proto_dims = proto_dims
        self.top_k = top_k
        self.num_samples = num_samples
        self.momentum = momentum
        self.tau = tau
        self.hard_sample = hard_sample
        self.loss_intra = loss_intra
        self.thresh = thresh
        self.proto_type = proto_type
        self.reg_norm = kwargs.pop('reg_norm', -114514)
        self.reg_weight = kwargs.pop('reg_weight', 1.0)
        super(ContrastiveHead, self).__init__(**kwargs)
        del self.conv_seg

        self.conv_proj = nn.Conv2d(
            self.in_channels,
            self.proto_dims,
            kernel_size=1,
            bias=False
        )
    
    def proj(self, inputs):
        x = self._transform_inputs(inputs)
        return self.conv_proj(x)

    def forward(self, inputs, seg_logits, domain):
        
        x = self._transform_inputs(inputs)
        v = self.conv_proj(x)

        b, d, h, w = v.shape
        
        label = F.interpolate(
            input=seg_logits.float(),
            size=(h, w),
            mode='nearest'
        ).long()
        v = v.permute(0, 2, 3, 1).contiguous().view(-1, self.proto_dims)
        label = label.view(-1)

        not_ignore_idx = torch.where(label < 255)
        v = v[not_ignore_idx]
        label = label[not_ignore_idx]
        v = F.normalize(v, p=2, dim=1)

        return v, label
    
    def forward_train(self, inputs, prototypes, seg_logits, domain):
        v, label = self.forward(inputs, seg_logits, domain)
        losses = self.losses(v, label, prototypes, domain)
        return losses


    def contrastive_loss(self, v, label, prototypes, domain):

        loss = dict()

        indices = torch.randperm(v.shape[0])[:self.num_samples]
        v_samples = v[indices]
        lbl_samples = label[indices]

        loss[f'{domain}_loss_cross'] = nn.CrossEntropyLoss()((v_samples @ prototypes.T) / self.tau, lbl_samples) * self.loss_weight

        return loss
    
    def reg_loss(self, v, prototypes, domain):
        v_mean = v.mean(axis=0, keepdim=True)

        loss = dict()
        
        logits = v_mean.mm(prototypes.detach().permute(1, 0)) / \
            self.tau
        loss[f'{domain}_reg_loss'] = \
            torch.sum(torch.softmax(logits, dim=1).log()) / self.reg_norm * self.reg_weight
        
        return loss



    @force_fp32(apply_to=('v_samples', ))
    def losses(self, v_samples, lbl_samples, prototypes, domain):
        loss = dict()
        if self.proto_type == 'centroid':
            loss.update(self.contrastive_loss(v_samples, lbl_samples, prototypes, domain))
        if self.reg_norm > 0:
            loss.update(self.reg_loss(v_samples, prototypes, domain))
        return loss


    def init_ema_weights(self, aux_head):
        for p in self.parameters():
            p.detach_()
        
        self.conv_proj.training = False
        
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
        
        
            




