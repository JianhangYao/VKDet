# Author: Yao
# CreatTime: 2025/3/25
# FileName: extract_img_attn_embeddings
# Description: simple introduction of the code
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import argparse
from mmdet.datasets.ASKD_dataset import DIORImage, DIORCroppedProposals, DOTAImage, DOTACroppedProposals
import random
import clip
import mmcv
from mmdet.apis import set_random_seed
from torch import nn
import time
from tqdm import tqdm
import os
import pickle
from mmdet.models.roi_heads.Adaptive_kd_weight import up_attn, attn_align, augment_box
from mmdet.our_utils.base_filter import filter_and_match_annotations
from mmdet.our_utils.filter_thr_p import filter_proposals

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '78920'
dist.init_process_group(backend='nccl', rank = 0, world_size = 1)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract DIOR embeddings")
    parser.add_argument("--dataset_type", default="DIOR", choices=["DIOR", "DOTA", "JQ"], help="dataset type")
    parser.add_argument("--data_root", default="data/DIOR/", help="data root")
    parser.add_argument("--split", default="train", help="data split")
    parser.add_argument("--mode", default="OVD", choices=["OVD", "CSD"], help="operation mode: OVD or CSD")
    parser.add_argument("--proposal_file", default="proposals/dior/filtered_train_proposals.pkl", help="path to pre-computed proposals")
    parser.add_argument("--clip_root", default="weights", help="clip model path")
    parser.add_argument("--num_workers", default=48, type=int, help="num workers per gpu") # 48
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu") # 128

    # ASKD parameterss
    parser.add_argument("--final_num_boxes", default=500, type=int, help="number of final bounding boxes to retain")
    parser.add_argument("--sharpen_temp", default=0.025, type=float, help="temperature parameter for attn sharpening")
    # PAPL parameters
    parser.add_argument("--fine_tune", default=False, type=bool, help="temperature parameter for attn sharpening")

    parser.add_argument("--save_enhance_proposals", default="askd_cache/DIOR_OVD_50_imgproposal.pkl", help="path to save output")
    parser.add_argument("--save_embed_path_pkl", default="askd_cache/DIOR_OVD_50_imgembed.pkl", help="path to save output")

    parser.add_argument("--local_rank",default=0, type=int)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--deterministic", default=True, help="whether to set deterministic options for CUDNN backend."
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = int(0)
    torch.cuda.set_device(device_id)

    # set random seeds
    if args.seed is not None:
        print(f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}")
        set_random_seed(args.seed, deterministic=args.deterministic)
    mode = args.mode

    # ASKD or PAPL
    if not args.fine_tune:
        print(f'======================== Start Adaptive Selective Knowledge Distillation ========================')
        mode = "OVD"
    else:
        print(f'======================== Start Prototype-Aware Pseudo-Label ========================')
        mode = "CSD"


    dataset = DIORImage(
        pipeline=[],
        cls_mode=mode,
        split_mode=args.split,
        data_root=args.data_root,
        img_prefix="JPEGImages-trainval",
        img_suffix='.jpg',
        filter_empty_gt=True
    )

    # Load CLIP
    model_name = 'ViT-B/32'
    clip_model, _ = clip.load(model_name, device=device_id, download_root=args.clip_root)
    ckpt_path = os.path.join(args.clip_root, "RemoteCLIP-ViT-B-32.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"CLIP weights not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    message = clip_model.load_state_dict(ckpt)
    print(message)

    clip_model.cuda().eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    # Build dataloaders
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, pin_memory=True)

    # Build informative attention hooks
    att_infor = extract_attention_maps(args, clip_model, dataloader, device)

    # Mask and Enhancer
    final_proposals = Mask_Enhancer(args, att_infor)

    # Reload crop dataloader
    crop_dataset = DIORCroppedProposals(
        pipeline=[],
        cls_mode=args.mode,
        split_mode=args.split,
        enhance_proposals=final_proposals,
        data_root=args.data_root,
        img_prefix="JPEGImages-trainval",
        img_suffix='.jpg',
        filter_empty_gt=True
    )
    crop_sampler = DistributedSampler(crop_dataset, world_size, rank, shuffle=False)
    crop_dataloader = data.DataLoader(crop_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=crop_sampler,
                                 pin_memory=True)
    embeddings = torch.HalfTensor(
        torch.HalfStorage.from_file(args.save_embed_path_pkl, shared=True, size=len(crop_dataset) * clip_model.visual.output_dim)
    ).reshape(len(crop_dataset), -1)
    full_embeddings = Croppedembed(args, clip_model, crop_dataloader, embeddings, device)

    # Save full image embeddings
    if rank == 0:
        features_per_image = []
        start_idx = 0
        proposals_per_img = crop_dataset.num_proposals_per_img[1:]

        for num in proposals_per_img:
            end_idx = start_idx + num
            img_features = full_embeddings[start_idx:end_idx].cpu().numpy().astype(np.float16)
            features_per_image.append(img_features)
            start_idx = end_idx

        features_dict = {}
        if len(crop_dataset.data_infos) != len(features_per_image):
            print(f"Warning: number of image infos ({len(crop_dataset.data_infos)}) does not match number of features ({len(features_per_image)}), possibly due to filtering empty annotations")
            # Clip to min length to avoid index overflow
            min_len = min(len(crop_dataset.data_infos), len(features_per_image))
            crop_dataset.data_infos = crop_dataset.data_infos[:min_len]
            features_per_image = features_per_image[:min_len]

        for img_info, feat in zip(crop_dataset.data_infos, features_per_image):
            img_name = img_info["img_name"]
            features_dict[img_name] = feat
        mmcv.dump(features_dict, args.save_embed_path_pkl)
        print(f"Feature dict saved to: {args.save_embed_path_pkl}")

def Croppedembed(args, clip_model, crop_dataloader, full_embeddings, device):
    print("======================== Start crop image encoding ========================")
    with torch.no_grad():
        for step, (img, img15, idx) in enumerate(tqdm(crop_dataloader, desc="Embedding Processing")):
            # Average crop-image embeddings over the receptive field
            img = img.to(device)
            img15 = img15.to(device)
            clip_image_features= clip_model.encode_image(img)
            clip_image_features15= clip_model.encode_image(img15)
            clip_image_features_single = clip_image_features + clip_image_features15
            clip_image_features1 = F.normalize(clip_image_features_single, p=2, dim=1)
            full_embeddings[idx] = clip_image_features1.cpu()

    return full_embeddings



def Mask_Enhancer(args, att_infor=None):
    # Load proposals
    with open(args.proposal_file, 'rb') as f:
        kd_proposals = pickle.load(f)

    kd_proposals, _ = filter_proposals(kd_proposals, 5) # drop proposals with w/h < 5
    print("======================== Start mask & enhance ========================")
    new_proposals = []
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
        # Mask
        attn = up_attn(att_infor['attentions'][i], args.sharpen_temp)
        instance_kd_weight = attn_align(args.data_root, filtered_1st, attn, att_infor['img_names'][i])
        instance_kd_weight = torch.tensor(instance_kd_weight).unsqueeze(1)
        filtered_1st = [torch.tensor(filtered_1st[i]) for i in range(len(filtered_1st))]
        selected_indices = np.where(instance_kd_weight)[0]
        filtered_2st = [filtered_1st[i].clone().detach() for i in selected_indices]

        # Enhance
        if 0 < len(filtered_2st) < args.final_num_boxes:
            required = args.final_num_boxes - len(filtered_2st)
            augmented = []
            while len(augmented) < required:
                base_box = random.choice(filtered_2st).tolist()
                x1, y1, x2, y2, score = base_box
                if np.abs(np.log10((x2 - x1) / (y2 - y1))) >= 1 :
                    augmented.append(augment_box(base_box, 0))
                    augmented.append(augment_box(base_box, 1)) # augment_box
                else:
                    augmented.append(augment_box(base_box, 1))
            filtered_2st += augmented

        # Base Filter
        if args.fine_tune:
            output = filter_and_match_annotations(att_infor, i, filtered_2st, args.data_root)
            if output:
                gt_instance_index = output['gt_instance_index'].any(dim=1, keepdim=True)
            else:
                gt_instance_index = torch.zeros(len(filtered_2st), dtype=torch.bool)
            gt_selected_indices = np.where(gt_instance_index)[0]
            non_gt_selected_indices = np.where(~gt_instance_index)[0]
            filtered_gt = [filtered_2st[i] for i in gt_selected_indices]
            filtered_nc = [filtered_2st[i] for i in non_gt_selected_indices]  # non-GT proposals act as negative correlation (NC)
            filtered_2st = filtered_nc
        else:
            filtered_2st = filtered_2st

        if len(filtered_2st) > 0:
            filtered_2st = torch.stack(filtered_2st)
            final_np = np.array(filtered_2st, dtype=np.float32)
        else:
            final_np = np.empty((0, 5), dtype=np.float32)
        new_proposals.append(final_np)


    # Save offline
    with open(args.save_enhance_proposals, 'wb') as f:
        pickle.dump(new_proposals, f)
    print(f"Proposals saved to: {args.save_enhance_proposals}")

    return new_proposals
def extract_attention_maps(args, model, dataloader, device):
    print("======================== Generating attention masks ========================")
    save_dict = {
        'embeddings': [],
        'attentions': [],
        'img_names': []
    }
    vision_model = model.visual
    num_layers = len(vision_model.transformer.resblocks)

    layer_attentions = {layer_idx: [] for layer_idx in range(num_layers)}
    def hook_fn_factory(layer_idx):
        def hook_fn(module, input, output):
            attn_weights = output[1].detach().cpu()
            layer_attentions[layer_idx].append(attn_weights)
        return hook_fn

    hooks = []
    # Register hooks on attention layers
    if hasattr(model.visual, 'transformer'):
        for layer_idx, block in enumerate(model.visual.transformer.resblocks):

            def make_forward(original_forward):
                def new_forward(query, key, value, **kwargs):
                    kwargs['need_weights'] = True
                    return original_forward(query, key, value, **kwargs)

                return new_forward

            # Save original forward and replace
            block.attn._original_forward = block.attn.forward
            block.attn.forward = make_forward(block.attn.forward)

            # Register hook
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
                    layer_attn = layer_attentions[layer_idx][batch_idx]  # [B, seq_len, seq_len]
                    img_layer_attn = layer_attn[img_in_batch_idx]  # [seq_len, seq_len]
                    cls_attn = img_layer_attn[0, 1:]  # [num_patches]

                    img_attentions.append(cls_attn)
                img_attentions = torch.stack(img_attentions)
                grid_size = int(np.sqrt(img_attentions[-1].shape))
                mean_layer_attn = torch.mean(img_attentions, dim=0, keepdims=True)
                if not isinstance(img_attentions, torch.Tensor):
                    mean_layer_attn = torch.tensor(mean_layer_attn, device=device)
                # mean_layer_attn = self.normalize_weights(mean_layer_attn)
                mean_layer_attn = mean_layer_attn.reshape(grid_size, grid_size)
                assert mean_layer_attn.shape == (7, 7), f"Invalid attention weight shape {mean_layer_attn.shape}"
                batch_attentions.append(mean_layer_attn)  # [num_layers, seq-1]
            save_dict['attentions'].extend(batch_attentions)

        for hook in hooks:
            hook.remove()

        final_data = {
            'embeddings': torch.cat(save_dict['embeddings'], dim=0),
            'attentions': torch.stack(save_dict['attentions']),
            'img_names': save_dict['img_names']
        }

        # Save offline

        return final_data

if __name__ == "__main__":
    main()