# Author: Yao
# CreatTime: 2024/11/27
# FileName: visdrone
# Description: simple introduction of the code
dataset_type = "XMLDataset_DOTA"
data_root = "data/DOTA1.0/"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.393, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadProposals", num_max_proposals=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "proposals", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadProposals", num_max_proposals=None),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size_divisor=32),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            # dict(type="Collect", keys=["img"]),
            dict(type="Collect", keys=["img", "proposals", "objectness"]), #"objectness", "proposals", "objectness"
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    min_size = 800,
    train=dict(
        type="XMLDataset_DOTA",
        ann_file="data/DOTA1.0/ImageSets/Main/cropped_images_train_filter_list.txt",
        proposal_file="proposals/dota/filtered_train/choose0.1_200_aug_train.pkl",
        img_prefix="data/DOTA1.0",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file="data/DOTA1.0/ImageSets/Main/cropped_images_train_filter_list.txt",
        proposal_file="proposals/dota/filtered_train/choose0.1_200_aug_train.pkl",
        img_prefix="data/DOTA1.0",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file="data/DOTA1.0/ImageSets/Main/cropped_images_val_list_with_objects.txt",
        proposal_file="proposals/dota/val/val_proposals_best.pkl",
        img_prefix="data/DOTA1.0",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="mAP")