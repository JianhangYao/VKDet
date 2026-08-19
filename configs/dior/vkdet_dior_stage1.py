import os.path as osp

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/dior_detection.py",
    "../_base_/schedules/schedule_20e.py",
    "../_base_/default_runtime.py",
]

data_root = "data/DIOR/"
weights_root = "weights"
cache_root = "askd_cache"
train_proposals = "proposals/dior/invalid_box_filter/filtered_train_proposals.pkl"
val_proposals = "proposals/dior/val/train_filtered_val_proposals.pkl"
text_embedding = "embeddings/dior/ovd_dior_text_embedding.pth"

# Stage 1 runs ASKD automatically. If matching disk caches exist, ASKD reuses
# them; remove the matching cache files to force recomputation.
prep = [
    dict(
        type="ASKD",
        dataset_type="DIOR",
        data_root=data_root,
        split="train",
        mode="OVD",
        fine_tune=False,
        proposal_file=train_proposals,
        clip_root=weights_root,
        cache_dir=cache_root,
        final_num_boxes=50,
        sharpen_temp=0.025,
        batch_size=64,
        num_workers=8,
        min_proposal_width=32,
        min_proposal_height=32,
        min_proposal_area=1024,
        seed=42,
    ),
]

fine_tune = False
load_from = "weights/current_mmdetection_Head.pth"

model = dict(
    pretrained=None,
    backbone=dict(frozen_stages=-1, norm_cfg=dict(type="BN", requires_grad=True), style="caffe"),
    neck=dict(norm_cfg=dict(type="BN", requires_grad=True)),
    roi_head=dict(
        type="StandardRoIHead",
        prompt_path=text_embedding,
        kd_weight=256,
        class_split=[16, 4],
        k=30,
        clip_root=weights_root,
        feature_path=osp.join(cache_root, "DIOR_OVD_50_imgembed.pkl"),
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            in_channels=256,
            ensemble=True,
            fc_out_channels=1024,
            roi_feat_size=7,
            with_cls=False,
            num_classes=16,
            norm_cfg=dict(type="BN", requires_grad=True),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=True,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1),
        ),
        temperature_test=0.01,
        temperature=0.1,
    ),
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[16, 19])
evaluation = dict(interval=20, metric=["mAP"])
checkpoint_config = dict(interval=5, create_symlink=True)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        mode="OVD",
        class_split=[16, 4],
        k=30,
        ann_file=osp.join(data_root, "ImageSets", "Main", "filtered_train.txt"),
        proposal_file=train_proposals,
        img_prefix=data_root,
    ),
    val=dict(
        mode="CSD",
        class_split=[16, 4],
        k=30,
        ann_file=osp.join(data_root, "ImageSets", "Main", "filtered_train.txt"),
        proposal_file=train_proposals,
        img_prefix=data_root,
    ),
    test=dict(
        mode="CSD",
        class_split=[16, 4],
        k=30,
        ann_file=osp.join(data_root, "ImageSets", "Main", "val.txt"),
        proposal_file=val_proposals,
        img_prefix=data_root,
    ),
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=15, norm_type=2))
total_epochs = 20
log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook")])
