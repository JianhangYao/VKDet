"""mmdet-embedded PAPL preprocessing.

DDP optimization: PAPL (clustering / faiss / json) is pure CPU, so it runs
once on rank 0 (all cores; avoids 8x redundancy + thread oversubscription),
then centers + json are broadcast to every rank. Rank 0's K_Means uses the
joblib threading backend (avoids fork deadlock inside DDP processes).
"""
import os
import shutil
import tempfile
import argparse
import pickle
import sys

import torch.distributed as dist

from mmdet.our_utils import prep_cache

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def run_papl_prep(cfg_prep, cfg=None, rank=0, world_size=1):
    # Persist PAPL artifacts for independent train/test processes. The input
    # pickle files remain temporary, but centers and pseudo-label JSON are
    # experiment outputs and must survive process termination.
    output_dir = cfg_prep.get('output_dir')
    if output_dir is None:
        work_dir = cfg.get('work_dir', 'workdirs/dior/step2') if cfg is not None else 'workdirs/dior/step2'
        output_dir = os.path.join(work_dir, 'prep')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_proj_root, output_dir)
    center_dir = os.path.join(output_dir, 'centers')
    json_dir = os.path.join(output_dir, 'json')
    os.makedirs(center_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    k_value = cfg_prep.get('k_value', 30)
    num_samples = cfg_prep.get('num_samples', 2000)
    centers_path = os.path.join(center_dir, f'{k_value}_cluster_centers.pth')
    json_path = os.path.join(json_dir, f'DIOR_vild_cluster{k_value}_proposal{num_samples}.json')

    if rank == 0:
        from VK_Det import PAPL

        imgembed = prep_cache.get('askd_imgembed')
        imgproposal = prep_cache.get('askd_imgproposal')
        if imgembed is None or imgproposal is None:
            raise RuntimeError(
                "[PAPL prep] prep_cache is missing ASKD outputs. Make sure ASKD runs before PAPL.")

        tmpdir = tempfile.mkdtemp(prefix='papl_prep_')
        feat_pkl = os.path.join(tmpdir, 'imgembed.pkl')
        prop_pkl = os.path.join(tmpdir, 'imgproposal.pkl')
        with open(feat_pkl, 'wb') as f:
            pickle.dump(imgembed, f)
        with open(prop_pkl, 'wb') as f:
            pickle.dump(imgproposal, f)
        args = argparse.Namespace(
            dataset_type='dior',
            PL_type=cfg_prep.get('PL_type', 'prototype'),
            data_root=cfg_prep['data_root'],
            feature_pkl_path=feat_pkl,
            proposals_pkl_path=prop_pkl,
            center_path=center_dir,
            text_embedding_path=cfg_prep.get('text_embedding_path', ''),
            num_base_classes=cfg_prep.get('num_base_classes', 16),
            num_samples=cfg_prep.get('num_samples', 2000),
            k_value=k_value,
            save_json_path=json_dir,
            local_rank=rank,
            seed=cfg_prep.get('seed', 42),
            deterministic=True,
            dynamic_k=cfg_prep.get('dynamic_k', False),
            top_p=cfg_prep.get('top_p', 2),
            t_sne=False,
            t_sne_gt=False,
            iou_thr=cfg_prep.get('iou_thr', 0.1),
            visualize_json=False,
        )

        print(f"[PAPL prep] rank0 start (k={args.k_value}, base={args.num_base_classes})")
        json_annotations, _sf, _sp, _sg, centers = PAPL.main(args)
        if not os.path.isfile(centers_path):
            raise RuntimeError(f'[PAPL prep] cluster centers were not written: {centers_path}')
        if not os.path.isfile(json_path):
            raise RuntimeError(f'[PAPL prep] pseudo-label JSON was not written: {json_path}')
        print(f"[PAPL prep] rank0 done: {len(centers)} centers")
        print(f"[PAPL prep] persistent centers: {centers_path}")
        print(f"[PAPL prep] persistent annotations: {json_path}")
        _results = [centers, json_annotations]
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        _results = [None, None]

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()
        dist.broadcast_object_list(_results, src=0)

    prep_cache.set('papl_centers', _results[0])
    prep_cache.set('papl_json', _results[1])
    prep_cache.set('papl_centers_path', centers_path)
    prep_cache.set('papl_json_path', json_path)
    if rank == 0:
        print(f"[PAPL prep] centers + json broadcast to all ranks -> cache")
    return _results[0]
