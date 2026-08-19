import os.path as osp

_base_ = "./vkdet_dior_stage1.py"

data_root = "data/DIOR/"
weights_root = "weights"
cache_root = "askd_cache"
text_embedding = "embeddings/dior/ovd_dior_text_embedding.pth"
papl_output_dir = "askd_cache/papl_dior_csd500"

prep = [
    dict(
        type="ASKD",
        dataset_type="DIOR",
        data_root=data_root,
        split="train",
        mode="CSD",
        fine_tune=True,
        proposal_file="proposals/dior/whole_train/whole_train_proposals.pkl",
        clip_root=weights_root,
        cache_dir=cache_root,
        final_num_boxes=500,
        sharpen_temp=0.025,
        batch_size=256,
        num_workers=32,
        min_proposal_width=32,
        min_proposal_height=32,
        min_proposal_area=1024,
        seed=42,
    ),
    dict(
        type="PAPL",
        data_root=data_root,
        text_embedding_path=text_embedding,
        num_base_classes=16,
        num_samples=5000,
        k_value=30,
        top_p=2,
        seed=42,
        output_dir=papl_output_dir,
    ),
]

fine_tune = True
load_from = "workdirs/dior/stage1/epoch_20.pth"

model = dict(
    type="FasterRCNNFreezeBackbone",
    pretrained=None,
    roi_head=dict(
        type="StandardRoIHeadFinetune",
        class_split=[16, 4],
        k=30,
        temperature_test=0.01,
        bbox_head=dict(num_classes=20),
        prompt_path=text_embedding,
        beta=0.5,
        alpha=0,
        centers_path=osp.join(papl_output_dir, "centers", "30_cluster_centers.pth"),
    ),
)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type="XMLNovel_json_Dataset",
        ann_file=osp.join(
            papl_output_dir, "json", "DIOR_vild_cluster30_proposal5000.json"
        ),
        proposal_file=osp.join(cache_root, "DIOR_CSD_500_imgproposal.pkl"),
    ),
    test=dict(mode="CSD", class_split=[16, 4], k=30),
)

optimizer = dict(lr=0.001)
total_epochs = 12
lr_config = dict(warmup_iters=20, warmup_ratio=0.1, step=[8, 11])
evaluation = dict(interval=12, metric=["mAP"])
checkpoint_config = dict(interval=4, create_symlink=True)
find_unused_parameters = True
log_config = dict(interval=5, hooks=[dict(type="TextLoggerHook")])
