import os.path as osp

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/dior_detection.py",
    "../_base_/schedules/schedule_20e.py",
    "../_base_/default_runtime.py",
]

prep = []

weights_root = "weights"
cache_root = "askd_cache"
text_embedding = "embeddings/dior/ovd_dior_text_embedding.pth"

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
            num_classes=20,
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

data = dict(
    test=dict(
        mode="CSD",
        class_split=[16, 4],
        k=30,
    ),
)
