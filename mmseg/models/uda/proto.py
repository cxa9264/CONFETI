from importlib.metadata import requires
from numpy import require, size
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prototypes(nn.Module):

    def __init__(
            self,
            num_classes=19,
            proto_dim=512,
            momentum=0.999,
            update_policy='ema',
            proto_type='centroid',
            ori_size=False,
            thresh=0.968):
        super(Prototypes, self).__init__()
        if update_policy not in ['ema', 'stats']:
            raise NotImplementedError

        self.proto_dim = proto_dim
        self.alpha = 1 - momentum
        self.update_policy = update_policy
        self.type = proto_type
        self.num_classes = num_classes
        self.ori_size = ori_size
        self.update_count = torch.zeros(self.num_classes)
        self.thresh = thresh

        self.register_buffer('proto', torch.zeros(self.num_classes,
                                                    self.proto_dim,
                                                    requires_grad=False))


    def to(self, device):
        self.proto = self.proto.to(device=device)


    def get_proto(self):
            return self.proto.detach()

    def update_prototypes(self,
                          v,
                          seg_logits,
                          cam_map,
                          domain_mask,
                          img_metas):

        if self.ori_size:
            v = F.interpolate(
                input=v,
                size=seg_logits.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            cam_map = F.interpolate(
                input=cam_map,
                size=seg_logits.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            label = seg_logits.reshape(-1)
        else:
            cam_map = F.interpolate(
                input=cam_map,
                size=v.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            if domain_mask is not None:
                domain_mask = F.interpolate(
                    input=domain_mask.float(),
                    size=v.shape[-2:],
                    mode='nearest',
                )
            label = F.interpolate(
                input=seg_logits.float(),
                size=v.shape[-2:],
                mode='nearest'
            ).long()
            label = label.reshape(-1)

        v = v.permute(0, 2, 3, 1).reshape(-1, self.proto_dim)
        cam_map = cam_map.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        cam_label = torch.argmax(cam_map, dim=-1)
        mask_ignore = (label != 255)
        if domain_mask is not None:
            domain_mask = domain_mask.reshape(-1)
            mask_ignore *= (domain_mask == 1)
        v = v[mask_ignore]
        cam_map = cam_map[mask_ignore]
        label = label[mask_ignore]
        cam_label = cam_label[mask_ignore]

        for i in label.unique():
            if self.update_count[i] == 0:
                alpha = 1
            else:
                alpha = min(1 - 1 / (self.update_count[i] + 1), self.alpha)
            mask_i = (label == i)
            feat_i = v[mask_i]

            w_i = cam_map[:, i][mask_i]
            assert w_i.shape[0] == feat_i.shape[0]
            pos = torch.argsort(w_i, descending=True)[:32]
            feat_i = feat_i[pos]
            w_i = w_i[pos].unsqueeze(0)
            self.update_count[i] += len(feat_i)
            if w_i.sum() == 0:
                continue
            feat_i = (w_i @ feat_i) / w_i.sum()

            feat_i = F.normalize(feat_i, p=2)
            self.proto[i, :] = self.proto[i, :] * (1 - alpha) + \
                alpha * feat_i
        
        self.proto = F.normalize(self.proto, p=2, dim=1)

   