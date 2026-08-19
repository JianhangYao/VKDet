"""mmdet-embedded ASKD preprocessing. A faithful port of VK_Det/ASKD.py
with three changes: 1) function-based (params from cfg.prep, no argparse);
2) no redundant dist init; 3) intermediate outputs go to the in-memory
prep_cache instead of disk pkl files.

CLIP loading follows the original openai-clip (clip.load + RemoteCLIP
load_state_dict) so attention-hook behavior matches the original exactly
(RemoteCLIP verified to have 0 missing keys vs openai-clip).
"""
import os
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import clip
from tqdm import tqdm

from mmdet.datasets.ASKD_dataset import DIORImage, DIORCroppedProposals
from mmdet.apis import set_random_seed
from mmdet.models.roi_heads.Adaptive_kd_weight import up_attn, attn_align, augment_box
from mmdet.our_utils.base_filter import filter_and_match_annotations
from mmdet.our_utils.filter_thr_p import filter_proposals
from mmdet.our_utils import prep_cache


def extract_attention_maps(args, model, dataloader, device):
    print("======================== Generating attention masks ========================")
    save_dict = {'embeddings': [], 'attentions': [], 'img_names': []}
    vision_model = model.visual
    num_layers = len(vision_model.transformer.resblocks)
    layer_attentions = {layer_idx: [] for layer_idx in range(num_layers)}

    def hook_fn_factory(layer_idx):
        def hook_fn(module, input, output):
            attn_weights = output[1].detach().cpu()
            layer_attentions[layer_idx].append(attn_weights)
        return hook_fn

    hooks = []
    if hasattr(model.visual, 'transformer'):
        for layer_idx, block in enumerate(model.visual.transformer.resblocks):
            def make_forward(original_forward):
                def new_forward(query, key, value, **kwargs):
                    kwargs['need_weights'] = True
                    return original_forward(query, key, value, **kwargs)
                return new_forward
            block.attn._original_forward = block.attn.forward
            block.attn.forward = make_forward(block.attn.forward)
            hook = block.attn.register_forward_hook(hook_fn_factory(layer_idx))
            hooks.append(hook)

    with torch.no_grad():
        for batch_idx, (imgs, _, img_names) in enumerate(tqdm(dataloader, desc="Attention Processing")):
            imgs = imgs.to(device)
            embeddings = model.encode_image(imgs).cpu()
            save_dict['embeddings'].append(embeddings)
            save_dict['img_names'].extend(img_names)
            batch_attentions = []
            for img_in_batch_idx in range(len(imgs)):
                img_attentions = []
                for layer_idx in range(num_layers):
                    layer_attn = layer_attentions[layer_idx][batch_idx]
                    img_layer_attn = layer_attn[img_in_batch_idx]
                    cls_attn = img_layer_attn[0, 1:]
                    img_attentions.append(cls_attn)
                img_attentions = torch.stack(img_attentions)
                grid_size = int(np.sqrt(img_attentions[-1].shape))
                mean_layer_attn = torch.mean(img_attentions, dim=0, keepdims=True)
                if not isinstance(img_attentions, torch.Tensor):
                    mean_layer_attn = torch.tensor(mean_layer_attn, device=device)
                mean_layer_attn = mean_layer_attn.reshape(grid_size, grid_size)
                assert mean_layer_attn.shape == (7, 7), f"Invalid attention weight shape {mean_layer_attn.shape}"
                batch_attentions.append(mean_layer_attn)
            save_dict['attentions'].extend(batch_attentions)

    for hook in hooks:
        hook.remove()
    for layer_idx, block in enumerate(model.visual.transformer.resblocks):
        if hasattr(block.attn, '_original_forward'):
            block.attn.forward = block.attn._original_forward

    final_data = {
        'embeddings': torch.cat(save_dict['embeddings'], dim=0),
        'attentions': torch.stack(save_dict['attentions']),
        'img_names': save_dict['img_names']
    }
    return final_data


def Croppedembed(args, clip_model, crop_dataloader, full_embeddings, device):
    print("======================== Start crop image encoding ========================")
    with torch.no_grad():
        for step, (img, img15, idx) in enumerate(tqdm(crop_dataloader, desc="Embedding Processing")):
            img = img.to(device)
            img15 = img15.to(device)
            clip_image_features = clip_model.encode_image(img)
            clip_image_features15 = clip_model.encode_image(img15)
            clip_image_features_single = clip_image_features + clip_image_features15
            clip_image_features1 = F.normalize(clip_image_features_single, p=2, dim=1)
            full_embeddings[idx] = clip_image_features1.cpu()
    return full_embeddings


def Mask_Enhancer(args, att_infor=None):
    with open(args.proposal_file, 'rb') as f:
        kd_proposals = pickle.load(f)
    print("======================== Start mask & enhance ========================")
    new_proposals = []
    filter_total = {
        'original': 0,
        'filtered': 0,
        'deleted': 0,
        'invalid_images': 0,
    }
    for i, tensor in enumerate(tqdm(kd_proposals, desc="Proposals Processing")):
        np_tensor = tensor.numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
        filtered_1st = [row for row in np_tensor if row[-1] > 0.1]
        if filtered_1st:
            sorted_arr = np.array(filtered_1st)
            sorted_arr = sorted_arr[sorted_arr[:, -1].argsort()[::-1]]
            num_boxes = min(args.final_num_boxes, len(sorted_arr))
            filtered_1st = [np.array(row) for row in sorted_arr[:num_boxes]]
        else:
            filtered_1st = []
        attn = up_attn(att_infor['attentions'][i], args.sharpen_temp)
        instance_kd_weight = attn_align(args.data_root, filtered_1st, attn, att_infor['img_names'][i])
        instance_kd_weight = torch.tensor(instance_kd_weight).unsqueeze(1)
        filtered_1st = [torch.tensor(filtered_1st[i]) for i in range(len(filtered_1st))]
        selected_indices = np.where(instance_kd_weight)[0]
        filtered_2st = [filtered_1st[i].clone().detach() for i in selected_indices]

        if 0 < len(filtered_2st) < args.final_num_boxes:
            required = args.final_num_boxes - len(filtered_2st)
            augmented = []
            while len(augmented) < required:
                base_box = random.choice(filtered_2st).tolist()
                x1, y1, x2, y2, score = base_box
                if np.abs(np.log10((x2 - x1) / (y2 - y1))) >= 1:
                    augmented.append(augment_box(base_box, 0))
                    augmented.append(augment_box(base_box, 1))
                else:
                    augmented.append(augment_box(base_box, 1))
            filtered_2st += augmented

        if args.fine_tune:
            output = filter_and_match_annotations(att_infor, i, filtered_2st, args.data_root)
            if output:
                gt_instance_index = output['gt_instance_index'].any(dim=1, keepdim=True)
            else:
                gt_instance_index = torch.zeros(len(filtered_2st), dtype=torch.bool)
            gt_selected_indices = np.where(gt_instance_index)[0]
            non_gt_selected_indices = np.where(~gt_instance_index)[0]
            filtered_gt = [filtered_2st[i] for i in gt_selected_indices]
            filtered_nc = [filtered_2st[i] for i in non_gt_selected_indices]
            filtered_2st = filtered_nc
        else:
            filtered_2st = filtered_2st

        if len(filtered_2st) > 0:
            filtered_2st = torch.stack(filtered_2st)
            final_np = np.array(filtered_2st, dtype=np.float32)
        else:
            final_np = np.empty((0, 5), dtype=np.float32)
        # Filter only after mask/enhance/fine-tune, before crop embedding.
        filtered_final, filter_stats = filter_proposals(
            [final_np],
            min_width=args.min_proposal_width,
            min_height=args.min_proposal_height,
            min_area=args.min_proposal_area,
        )
        new_proposals.append(filtered_final[0])
        for key in filter_total:
            filter_total[key] += filter_stats[key]

    print(
        "[ASKD] final proposal filter (after mask/enhance): "
        f"{filter_total['original']} -> {filter_total['filtered']} "
        f"(deleted {filter_total['deleted']}, "
        f"invalid_images={filter_total['invalid_images']}, "
        f"min_width={args.min_proposal_width}, "
        f"min_height={args.min_proposal_height}, "
        f"min_area={args.min_proposal_area})"
    )

    # Embedded version: keep in memory (original pickles to save_enhance_proposals)
    return new_proposals


def run_askd_prep(cfg_prep, cfg=None, rank=0, world_size=1):
    args = argparse.Namespace(
        dataset_type=cfg_prep.get('dataset_type', 'DIOR'),
        data_root=cfg_prep['data_root'],
        split=cfg_prep.get('split', 'train'),
        mode=('CSD' if cfg_prep.get('fine_tune', False) else 'OVD'),
        proposal_file=cfg_prep['proposal_file'],
        clip_root=cfg_prep['clip_root'],
        num_workers=cfg_prep.get('num_workers', 8),
        batch_size=cfg_prep.get('batch_size', 32),
        final_num_boxes=cfg_prep.get('final_num_boxes', 50),
        sharpen_temp=cfg_prep.get('sharpen_temp', 0.025),
        fine_tune=cfg_prep.get('fine_tune', False),
        min_proposal_width=cfg_prep.get('min_proposal_width', 32.0),
        min_proposal_height=cfg_prep.get('min_proposal_height', 32.0),
        min_proposal_area=cfg_prep.get('min_proposal_area', 1024.0),
        seed=cfg_prep.get('seed', 42),
        deterministic=True,
    )

    cache_dir = cfg_prep.get('cache_dir', 'askd_cache')
    os.makedirs(cache_dir, exist_ok=True)
    tag = "{}_{}_{}".format(args.dataset_type, args.mode, args.final_num_boxes)
    _emb_file = os.path.join(cache_dir, tag + "_imgembed.pkl")
    _prop_file = os.path.join(cache_dir, tag + "_imgproposal.pkl")
    if os.path.exists(_emb_file) and os.path.exists(_prop_file):
        prep_cache.set('askd_imgembed', pickle.load(open(_emb_file, 'rb')))
        prep_cache.set('askd_imgproposal', pickle.load(open(_prop_file, 'rb')))
        if rank == 0:
            print("[ASKD prep] HIT disk cache, skip compute: " + tag)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    if not args.fine_tune:
        print('======================== Start Adaptive Selective Knowledge Distillation ========================')
        mode = "OVD"
    else:
        print('======================== Start Prototype-Aware Pseudo-Label ========================')
        mode = "CSD"

    dataset = DIORImage(
        pipeline=[], cls_mode=mode, split_mode=args.split, data_root=args.data_root,
        img_prefix="JPEGImages-trainval", img_suffix='.jpg', filter_empty_gt=True
    )

    model_name = 'ViT-B/32'
    clip_model, _ = clip.load(model_name, device=device_id, download_root=args.clip_root)
    ckpt_path = os.path.join(args.clip_root, "RemoteCLIP-ViT-B-32.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"CLIP weights not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    clip_model.load_state_dict(ckpt)
    clip_model.cuda().eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    # DDP: every rank independently computes the FULL cache (world_size=1 sampling)
    # so each rank's prep_cache is complete (distillation needs full imgembed).
    # 8 GPUs each compute a copy; wall-time ~= single GPU.
    _ws, _rk = 1, 0
    sampler = DistributedSampler(dataset, _ws, _rk, shuffle=False)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 sampler=sampler, pin_memory=True)

    att_infor = extract_attention_maps(args, clip_model, dataloader, device)
    final_proposals = Mask_Enhancer(args, att_infor)

    # proposals -> in-memory cache (replaces disk pkl)
    prep_cache.set('askd_imgproposal', final_proposals)
    if rank == 0:
        print(f"[ASKD prep] proposals -> cache: {len(final_proposals)} imgs")

    crop_dataset = DIORCroppedProposals(
        pipeline=[], cls_mode=args.mode, split_mode=args.split, enhance_proposals=final_proposals,
        data_root=args.data_root, img_prefix="JPEGImages-trainval", img_suffix='.jpg', filter_empty_gt=True
    )
    crop_sampler = DistributedSampler(crop_dataset, _ws, _rk, shuffle=False)
    crop_dataloader = data.DataLoader(crop_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      sampler=crop_sampler, pin_memory=True)

    # Embedded version: in-memory tensor replaces HalfStorage.from_file mmap
    out_dim = clip_model.visual.output_dim
    embeddings = torch.zeros(len(crop_dataset), out_dim, dtype=torch.half)
    full_embeddings = Croppedembed(args, clip_model, crop_dataloader, embeddings, device)

    # DDP: every rank must build its own imgembed cache (rank-0-only would leave
    # other ranks reading an empty cache and failing on disk fallback). The
    # original's rank==0 guard exists only to dedupe disk-file writes.
    features_per_image = []
    start_idx = 0
    proposals_per_img = crop_dataset.num_proposals_per_img[1:]
    for num in proposals_per_img:
        end_idx = start_idx + num
        img_features = full_embeddings[start_idx:end_idx].cpu().numpy().astype(np.float16)
        features_per_image.append(img_features)
        start_idx = end_idx

    features_dict = {}
    n = min(len(crop_dataset.data_infos), len(features_per_image))
    for i in range(n):
        features_dict[crop_dataset.data_infos[i]['img_name']] = features_per_image[i]

    prep_cache.set('askd_imgembed', features_dict)
    if rank == 0:
        pickle.dump(final_proposals, open(_prop_file, 'wb'))
        pickle.dump(features_dict, open(_emb_file, 'wb'))
        print("[ASKD prep] saved to disk cache: " + tag)
        print(f"[ASKD prep] imgembed -> cache: {len(features_dict)} imgs")
    return final_proposals
