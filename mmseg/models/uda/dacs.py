# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS

import math
import os
import io
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mmseg.ops.wrappers import resize
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from mmcv.runner import load_checkpoint
from PIL import Image

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from ..builder import build_head
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, renorm,
                                                get_mean_std, renorm_, strong_transform,
                                                renorm01, denorm01)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from ..decode_heads.cam_head import CAMHead
from ..decode_heads.contrastive_head import ContrastiveHead
from .proto import Prototypes
from cut.models import create_model
from cut.options.train_options import TrainOptions
from ..utils.sepico_transforms import RandomCrop, RandomCropNoProd
from torchvision.transforms import RandomCrop as TorchRandomCrop
from .fda import FDA_source_to_target

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

def set_requires_grad(module, val):
  for p in module.parameters():
    p.requires_grad = val

@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.checkpoint = cfg.get('checkpoint', None)
        self.proda = cfg.get('proda', False)
        self.multi_layer = cfg.get('multi_layer', False)
        self.proto_dims = cfg.get('proto_dims', None)
        self.in_dims = cfg.get('in_dims', None)
        self.exp_name = cfg.get('exp_name', '')
        self.enable_cbc = cfg.get('enable_cbc', False)
        self.cat_max_ratio = cfg.get('cat_max_ratio', 0.75)
        self.train_cut = cfg.get('train_cut', False)
        self.enable_cut = cfg.get('enable_cut', False)
        assert self.mix == 'class'

        if 'aux_heads' in cfg.keys():
            self.aux_heads_cfg = cfg['aux_heads']
        else:
            self.aux_heads_cfg = None

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        if self.aux_heads_cfg is not None:
            if isinstance(self.aux_heads_cfg, list):

                if self.multi_layer:
                    self.aux_heads = nn.ModuleList()
                    self.ema_aux_heads = nn.ModuleList()
                    for i, (in_dim, proto_dim) in enumerate(zip(self.in_dims, self.proto_dims)):
                        for aux_head_cfg in self.aux_heads_cfg:
                            aux_head_cfg['in_index'] = i
                            aux_head_cfg['in_channels'] = in_dim
                            if aux_head_cfg['type'] == 'ContrastiveHead':
                                aux_head_cfg['proto_dims'] = proto_dim
                            ema_aux_head_cfg = deepcopy(aux_head_cfg)
                            self.aux_heads.append(build_head(deepcopy(aux_head_cfg)))
                            self.ema_aux_heads.append(build_head(ema_aux_head_cfg))
                else:
                    self.aux_heads = nn.ModuleList()
                    self.ema_aux_heads = nn.ModuleList()
                    for aux_head_cfg in self.aux_heads_cfg:
                        ema_aux_head_cfg = deepcopy(aux_head_cfg)
                        self.aux_heads.append(build_head(deepcopy(aux_head_cfg)))
                        self.ema_aux_heads.append(build_head(ema_aux_head_cfg))
            else:
                self.aux_heads = build_head(deepcopy(self.aux_heads_cfg))
                self.ema_aux_heads = build_head(deepcopy(self.aux_heads_cfg))
            

            if self.multi_layer:
                self.src_proto = nn.ModuleList()
                self.tgt_proto = nn.ModuleList()
                for proto_dim in self.proto_dims:
                    proto_cfg = deepcopy(cfg['proto'])
                    proto_cfg['proto_dim'] = proto_dim
                    self.src_proto.append(Prototypes(**proto_cfg))
                    self.tgt_proto.append(Prototypes(**proto_cfg))
            else:
                assert 'proto' in cfg.keys(), 'no prototypes config'
                self.tgt_proto = Prototypes(**cfg['proto'])
        else:
            self.aux_heads = None
            self.ema_aux_heads = None

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        
        # Style transfer model
        if self.enable_cut:
            self.opt = TrainOptions().parse()
            self.cut_model = create_model(self.opt)
            self.reg_proj = nn.Conv2d(512,
                        512,
                        kernel_size=1,
                        stride=1,
                        padding=0)
                    
    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def extract_ema_cl_feat(self, img):
        feat = self.get_ema_model().extract_feat(img)
        x = self.ema_aux_heads[1].proj(feat, cut=False)
        return x

    def random_crop(self, image, gt_seg, prod=True):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        image = image.permute(0, 2, 3, 1).contiguous()
        gt_seg = gt_seg
        res_img, res_gt = [], []
        for img, gt in zip(image, gt_seg):
            results = {'img': img, 'gt_semantic_seg': gt, 'seg_fields': ['gt_semantic_seg']}
            results = RC(results)
            img, gt = results['img'], results['gt_semantic_seg']
            res_img.append(img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        return image, gt_seg

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

        if self.aux_heads is not None:
            if isinstance(self.aux_heads, nn.ModuleList):
                for ema_aux_head, aux_head in zip(self.ema_aux_heads, self.aux_heads):
                    ema_aux_head.init_ema_weights(aux_head)
            else:
                self.ema_aux_heads.init_ema_weights(self.aux_heads)

    def _update_ema(self, iter, proda=False):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        if not proda:
            for ema_param, param in zip(self.get_ema_model().parameters(),
                                        self.get_model().parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else:
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]

        if self.aux_heads is not None:
            if isinstance(self.aux_heads, nn.ModuleList):
                for ema_aux_head, aux_head in zip(self.ema_aux_heads, self.aux_heads):
                    ema_aux_head.update_ema_weights(aux_head, alpha_teacher)
            else:
                self.ema_aux_heads.update_ema_weights(
                    self.aux_heads, alpha_teacher)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def gan_step(self):
        # update D
        self.cut_model.set_requires_grad(self.cut_model.netD, True)
        self.cut_model.loss_D = self.cut_model.compute_D_loss()
        self.cut_model.loss_D.backward()
        self.cut_model.optimizer_D.step()
        self.cut_model.optimizer_D.zero_grad()

        # update G
        self.cut_model.set_requires_grad(self.cut_model.netD, False)
        self.cut_model.loss_G = self.cut_model.compute_G_loss()
        self.cut_model.loss_G.backward()
        self.cut_model.optimizer_G.step()
        self.cut_model.optimizer_G.zero_grad()
        if self.cut_model.opt.netF == 'mlp_sample':
            self.cut_model.optimizer_F.step()
            self.cut_model.optimizer_F.zero_grad()

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),
            'std': stds[0].unsqueeze(0)
        }


        if self.enable_cut:
            s2t_means_stds = torch.tensor([0.5, 0.5, 0.5], device=dev).reshape(1, 3, 1, 1)
            means01, stds01 = means / 255, stds / 255

            s2t_A_img = renorm01(denorm01(img, means01, stds01), s2t_means_stds, s2t_means_stds)
            s2t_B_img = renorm01(denorm01(target_img, means01, stds01), s2t_means_stds, s2t_means_stds)

            data = dict()
            data['A'] = s2t_A_img 
            data['B'] = s2t_B_img 
            data['A_paths'] = ""
            data['B_paths'] = ""

        # Init/update ema model
        if self.local_iter == 0:

            if self.enable_cbc:
                self.crop_size = s2t_A_img.shape[-2:]
            
            if self.enable_cut:
                if not self.train_cut:
                    self.cut_model.opt.isTrain = False
                    self.cut_model.isTrain = False
                self.cut_model.data_dependent_initialize(data)
                self.cut_model.setup(self.opt, not self.train_cut)
                if not self.train_cut:
                    set_requires_grad(self.cut_model.netG, False)
                    self.cut_model.eval()

            self._init_ema_weights()
            if self.aux_heads is not None:
                self.tgt_proto.to(dev)
            

        if self.local_iter > 0:
            self._update_ema(self.local_iter, proda=self.proda)


        # Style Transfer loss
        if self.enable_cut:
            self.cut_model.set_input(data)
            if self.train_cut:
                self.cut_model.forward()
                s2t_img = self.cut_model.fake_B
            else:
                s2t_img = self.cut_model.netG(data['A'])
            s2t_img = renorm01(denorm01(s2t_img, s2t_means_stds, s2t_means_stds), means01, stds01).to(dev)
        elif self.enable_fda:
            s2t_img = FDA_source_to_target(img, target_img, 0.09).detach()
        else:
            s2t_img = img


        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        if not self.enable_cbc:
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=dev)

        if self.enable_cbc:
            target_img, pseudo_label = self.random_crop(target_img, pseudo_label)
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_label.shape, device=dev)
        
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((s2t_img[i].clone().detach(), target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        mix_masks = torch.cat(mix_masks)

        if self.aux_heads is not None:
            with torch.no_grad():
                src_ema_feat = self.get_ema_model().extract_feat(s2t_img)
                mix_ema_feat = self.get_ema_model().extract_feat(mixed_img)
                if self.ema_aux_heads is not None:
                    src_ema_cam_map, src_ema_cam_scores = \
                        self.ema_aux_heads[0].forward_test(
                            src_ema_feat, None, None)
                    mix_ema_cam_map, mix_ema_cam_scores = \
                        self.ema_aux_heads[0].forward_test(
                            mix_ema_feat, None, None)

                    self.tgt_proto.update_prototypes(
                        self.ema_aux_heads[1].proj(mix_ema_feat),
                        mixed_lbl,
                        # mix_ema_logits.detach(),
                        mix_ema_cam_map.detach() if self.local_iter > 3000 else torch.ones_like(src_ema_cam_map.detach(), device=dev),
                        None if self.local_iter > 3000 else mix_masks,
                        img_metas
                    )

        # Train on source images
        clean_losses = self.get_model().forward_train(
            s2t_img.clone().detach(), img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        if self.aux_heads is not None: 
            src_map = None
            for aux_head in self.aux_heads:
                if isinstance(aux_head, CAMHead):
                    src_aux_loss, f = aux_head.forward_train(
                        src_feat, img_metas, gt_semantic_seg)
                    src_map = f['cam_map']
                    src_scores = f['cam_scores']
                elif isinstance(aux_head, ContrastiveHead):
                    if self.local_iter > 3000:  # and not self.train_cut:
                        prototypes = self.tgt_proto.get_proto()
                        src_aux_loss = aux_head.forward_train(
                            inputs=src_feat,
                            seg_logits=gt_semantic_seg,
                            domain='src',
                            prototypes=prototypes
                        )

                    # gan
                    if self.train_cut:
                        set_requires_grad(self.get_model().backbone, False)
                        set_requires_grad(aux_head, False)

                        ori_src_feat = self.get_model().extract_feat(img)
                        ori_src_feat = aux_head.proj(ori_src_feat)

                        ori_src_feat = ori_src_feat.permute(0, 2, 3, 1)
                        src_sim = ori_src_feat @ self.tgt_proto.get_proto().T

                        s2t_feat = self.get_model().extract_feat(s2t_img)
                        s2t_feat = aux_head.proj(s2t_feat)

                        s2t_feat = self.reg_proj(s2t_feat)
                        s2t_feat = s2t_feat.permute(0, 2, 3, 1)
                        s2t_sim = s2t_feat @ self.tgt_proto.get_proto().T

                        gan_loss = dict()
                        gan_loss['gan_consis_loss'] = \
                            nn.MSELoss(reduction='mean')(
                                s2t_sim, src_sim.detach()) * 0.1

                        clean_losses.update(gan_loss)

                        set_requires_grad(self.get_model().backbone, True)
                        set_requires_grad(aux_head, True)

                clean_losses.update(src_aux_loss)

        

        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')
        
       

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward() #
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        
        if self.train_cut:
            self.gan_step()

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_feat = mix_losses.pop('features')
        if self.aux_heads is not None and self.local_iter > 2000:
            mix_map = None
            for aux_head in self.aux_heads:
                if isinstance(aux_head, CAMHead):
                    mix_aux_loss, f = aux_head.forward_train(
                        mix_feat, img_metas, mixed_lbl)
                    mix_map = f['cam_map']
                    mix_scores = f['cam_scores']
                elif isinstance(aux_head, ContrastiveHead):
                    if self.local_iter < 3000: 
                        continue
                    prototypes = self.tgt_proto.get_proto()
                    mix_aux_loss = aux_head.forward_train(
                        inputs=mix_feat,
                        seg_logits=mixed_lbl,
                        domain='mix',
                        prototypes=prototypes
                    )
                mix_losses.update(mix_aux_loss)
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        if self.local_iter % 4000 == 0 and self.train_cut:
            if not os.path.exists('cut_checkpoints'):
                os.mkdir('cut_checkpoints')
            path = 'cut_checkpoints/%s_netG_%d.pth'%(self.exp_name, self.local_iter)
            torch.save(self.cut_model.netG.state_dict(), path)

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_s2t_img = torch.clamp(denorm(s2t_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')

                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                subplotimg(
                    axs[0][4], vis_s2t_img[j], 's2t image')

                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()

        self.local_iter += 1

        return log_vars