import os.path as osp

# Author: Yao
# CreatTime: 2024/11/27
# FileName: visdrone
# Description: simple introduction of the code
dataset_type = "JQ_Dataset"
data_root = "data/JQ_O/xml_type/"
jq_proposal_root = "proposals/jq"

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
            dict(type="Collect", keys=["img", "proposals", "objectness"]), # , "proposals", "objectness"
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        mode="OVD",
        class_split=[11, 4],
        k=30,  # num unknown classes
        ann_file=osp.join(data_root, "ImageSets", "Main", "filtered_train.txt"),
        proposal_file=osp.join(jq_proposal_root, "imgproposal_filtered_train.pkl"),
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        mode="CSD",
        class_split=[11, 4],
        k=30,  # num unknown classes
        ann_file=osp.join(data_root, "ImageSets", "Main", "filtered_train.txt"),
        proposal_file=osp.join(jq_proposal_root, "imgproposal_filtered_train.pkl"),
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        mode="CSD",
        class_split=[11, 4],
        k=30,  # num unknown classes
        ann_file=osp.join(data_root, "ImageSets", "Main", "val.txt"),
        proposal_file=osp.join(jq_proposal_root, "val_proposals.pkl"),
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="mAP")
