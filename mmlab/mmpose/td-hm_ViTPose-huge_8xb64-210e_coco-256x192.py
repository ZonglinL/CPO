# cp data/annotations/person_keypoints_val2017.json work_dirs/eval_dirs/HumanArt/lllyasviel_control_v11p_sd15_openpose_7.5-20/
# mim test mmpose mmlab/mmpose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py --checkpoint https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-huge_8xb64-210e_humanart-256x192-603bb573_20230612.pth --gpus 4 --launcher pytorch

data_root='work_dirs/eval_dirs/Captioned_COCOPose/lllyasviel_control_v11p_sd15_openpose_7.5-20'
work_dir=f'work_dirs/eval/mmpose/td-hm_ViTPose-huge_8xb64-210e_coco-256x192/{data_root.split("/")[-1]}'

auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
codec = dict(
    heatmap_size=(
        48,
        64,
    ),
    input_size=(
        192,
        256,
    ),
    sigma=2,
    type='UDPHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
    ])
data_mode = 'topdown'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        arch='huge',
        drop_path_rate=0.55,
        img_size=(
            256,
            192,
        ),
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_huge.pth',
            type='Pretrained'),
        patch_cfg=dict(padding=2),
        patch_size=16,
        qkv_bias=True,
        type='mmpretrain.VisionTransformer',
        out_type="featmap",
        with_cls_token=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='UDPHeatmap'),
        deconv_kernel_sizes=(
            4,
            4,
        ),
        deconv_out_channels=(
            256,
            256,
        ),
        in_channels=1280,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=17,
        type='HeatmapHead'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=False),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    constructor='LayerDecayOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0005, type='AdamW', weight_decay=0.1),
    paramwise_cfg=dict(
        custom_keys=dict(
            bias=dict(decay_multi=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)),
        layer_decay_rate=0.85,
        num_layers=32))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            170,
            200,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()

concated_dataset = [
    dict(
        ann_file='validation_coco.json',
        data_mode='topdown',
        data_prefix=dict(
            img=f'images/group_{idx}/coco/val2017'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'
    ) for idx in range(4)
]

test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='ConcatDataset',
        datasets=concated_dataset
    ),
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='validation_coco.json',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='HumanArt/annotations/training_humanart_coco.json',
        data_mode='topdown',
        data_prefix=dict(img=''),
        data_root='data/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(
                encoder=dict(
                    heatmap_size=(
                        48,
                        64,
                    ),
                    input_size=(
                        192,
                        256,
                    ),
                    sigma=2,
                    type='UDPHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(
        encoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type='UDPHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_mode='topdown',
        data_prefix=dict(img=''),
        data_root='data/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine', use_udp=True),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/annotations/person_keypoints_val2017.json',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine', use_udp=True),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
