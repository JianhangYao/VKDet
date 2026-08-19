import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

import argparse
import copy
import os
import os.path as osp
import time
import warnings
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "--config",
        default="configs/dior/vkdet_dior_stage2.py",
        help="train config file path",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="directory for logs/checkpoints; defaults to config or work_dirs/<config>",
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--fine-tune",
        "--fine_tune",
        dest="fine_tune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the config fine_tune flag",
    )
    parser.add_argument(
        "--no-validate", action="store_true", help="whether not to evaluate the checkpoint during training"
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus", type=int, help="number of gpus to use " "(only applicable to non-distributed training)"
    )
    group_gpus.add_argument(
        "--gpu-ids",
        "--gpu_ids",
        dest="gpu_ids",
        type=int,
        nargs="+",
        help="GPU ids for non-distributed training",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable deterministic CUDNN behavior (use --no-deterministic to disable)",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none", help="job launcher")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])
    if args.fine_tune is not None:
        cfg.fine_tune = args.fine_tune
    elif cfg.get("fine_tune", None) is None:
        cfg.fine_tune = False
    # auto resume, scan existing file
    # if os.path.exists(cfg.work_dir):
    #     work_dir_files = os.listdir(cfg.work_dir)
    #     work_dir_files = [f for f in work_dir_files if f.endswith(".pth") and f.split('.')[0] != 'latest']
    #     if len(work_dir_files)!=0:
    #         work_dir_files = sorted(work_dir_files, key=lambda y: int(y.split('.')[0].split('_')[-1]), reverse=True)
    #         resume_file = work_dir_files[0]
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    # auto resume
    # if args.resume_from is None and resume_file != "":
    # cfg.resume_from = os.path.join(cfg.work_dir, resume_file)
    # print("Auto resume from {}".format(cfg.resume_from))

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info(f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}")
        set_random_seed(args.seed, deterministic=args.deterministic)
    # Force deterministic options
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    # prep stage: if cfg has `prep`, run ASKD/PAPL preprocessing before building
    # the model/dataset; outputs go to the process-level in-memory cache
    # (mmdet.our_utils.prep_cache).
    if getattr(cfg, "prep", None):
        rank, world_size = get_dist_info()
        for step in cfg.prep:
            step = dict(step)
            step_type = step.pop("type", None)
            logger.info(f"[prep] running {step_type}: {step}")
            if step_type == "ASKD":
                from mmdet.our_utils import askd_prep
                askd_prep.run_askd_prep(step, cfg=cfg, rank=rank, world_size=world_size)
            elif step_type == "PAPL":
                from mmdet.our_utils import papl_prep
                papl_prep.run_papl_prep(step, cfg=cfg, rank=rank, world_size=world_size)
            else:
                raise ValueError(f"Unknown prep type: {step_type}")

        # PAPL's JSON pseudo-labels feed XMLNovel_json_Dataset (COCO API needs
        # a file): write the cache to a temp json and point ann_file at it,
        # keeping the "single command, no persistent pkl" flow.
        from mmdet.our_utils import prep_cache
        import json as _json
        _papl_json = prep_cache.get('papl_json')
        if _papl_json is not None:
            import tempfile as _tf
            _tmpj = _tf.NamedTemporaryFile('w', suffix='.json', delete=False)
            _json.dump(_papl_json, _tmpj); _tmpj.close()
            cfg.data.train.ann_file = _tmpj.name
            logger.info(f"[prep] PAPL json -> temp ann_file: {_tmpj.name}")

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.fine_tune:
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].OVD_Unknown_CLS)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].OVD_Unknown_CLS
    else:
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES


    train_detector(
        model, datasets, cfg, distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta
    )


if __name__ == "__main__":
    main()
