# optimizer
# optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[7000, 9000])
runner = dict(
    type='IterBasedRunner',  # Key change: EpochBased -> IterBased
    max_iters=10000)