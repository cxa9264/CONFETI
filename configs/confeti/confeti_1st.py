_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv2_r50-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 114
name = 'gta2cs_cut_2nd'
# Modifications to Basic UDA
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        depth=101)
)
cam_head = dict(
    type='CAMHead',
    in_channels=2048,
    channels=512,
    in_index=3,
    num_classes=19,
    init_cfg=dict(
        type='Normal', std=0.01),
)
contrastive_head = dict(
    type='ContrastiveHead',
    in_channels=2048,
    channels=512,
    in_index=3,
    num_classes=19,
    top_k=32,
    num_samples=4096,
    momentum=0.001,
    loss_weight=0.5,
    proto_dims=512,
    tau=0.1,
    reg_norm=55.94434060416236,
    init_cfg=dict(type='Normal', std=0.01),
    reg_weight=2
)
proto = dict(
    num_classes=19,
    proto_dim=512,
    momentum=0.999,
    update_policy='ema',
    proto_type='centroid',
    ori_size=True,
)
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    aux_heads=[cam_head, contrastive_head],
    proto=proto,
    exp_name=name,
    enable_cut=True,
    train_cut=True,
)
data = dict(
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))

# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-5,
    paramwise_cfg=dict(
        custom_keys=dict(
            cut_model=dict(lr_mult=0.0),
            head=dict(lr_mult=10.0),
            reg_proj=dict(lr_mult=10.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=80000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=2)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
# Meta Information for Result Analysis
exp = 'gta2cs_deeplabv2_r101_d8_confetti_1st'