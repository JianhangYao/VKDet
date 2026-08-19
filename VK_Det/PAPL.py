# Author: Yao
# CreatTime: 2025/10/14
# FileName: PAPL
# Description:
# For DOTA: data_root = "<DOTA>/train/JPEGImages-train"
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import os
import pickle
from sklearn.preprocessing import StandardScaler
import faiss
from PIL import Image, ImageFont
import json
from mmdet.models.roi_heads.class_name import DIOR_CLASSES, DOTA_CLASSES, JQ_CLASSES
from tqdm import tqdm
from mmdet.our_utils import K_Means, visualize_tsne, visualize_from_json, generate_gt_annotations_from_xml
from mmdet.our_utils.iou_compute import calculate_iou
from mmdet.apis import set_random_seed
import cv2
import time



if __name__ == '__main__' and not dist.is_initialized():
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group(backend='nccl', rank=0, world_size=1)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract pseudo-labels and generate cluster centers (DIOR)")
    parser.add_argument("--dataset_type", type=str, default="JQ", choices={"dior", "dota", "JQ"}, help="config file path")
    parser.add_argument("--PL_type", default="prototype", choices=["prototype", "text"], type=str, help="type of proposals")
    parser.add_argument("--data_root", default="data/DIOR", type=str, help="path to dataset")
    parser.add_argument("--feature_pkl_path", default="askd_cache/DIOR_OVD_50_imgembed.pkl", type=str, help="image embedding path")
    parser.add_argument("--proposals_pkl_path", default="askd_cache/DIOR_OVD_50_imgproposal.pkl", type=str, help="proposals path")
    parser.add_argument("--center_path", default="work_dirs/papl", type=str, help="center path")
    parser.add_argument("--text_embedding_path", default="text_embeddings/dior_novel_text_embedding.pth", type=str, help="novel text embeddings path")
    parser.add_argument("--num_base_classes", default=11, type=int, help="num base classes")
    parser.add_argument("--num_samples", default=2000, type=int, help="save_number of samples")
    parser.add_argument("--k_value", default=30, type=int, help="k_value")
    # Output paths
    parser.add_argument("--save_json_path", default="work_dirs/papl/json_data", type=str, help="save json file")
    parser.add_argument("--local_rank",default=0, type=int)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--deterministic", default=True, help="whether to set deterministic options for CUDNN backend.")
    # Analysis tools
    parser.add_argument("--dynamic_k", default=False, action="store_true", help="whether to dynamic k value")
    parser.add_argument("--top_p", default=2, type=int, help="num of prototypes")
    parser.add_argument("--t_sne", default=True, type=bool, help="Tsne or not")
    parser.add_argument("--t_sne_gt", default=True, type=bool, help="Tsne gt or not")
    parser.add_argument("--iou_thr", default=0.1, type=float, help="iou threshold")
    parser.add_argument("--visualize_json", default=False, type=bool, help="visualize_json or not")

    args = parser.parse_args()
    return args



def main(args):

    # Load precomputed embeddings
    with open(args.feature_pkl_path, 'rb') as f:
        clip_features = pickle.load(f)

    if args.seed is not None:
        print(f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}")
        set_random_seed(args.seed, deterministic=args.deterministic)

    if args.PL_type == "prototype":
        print(f"\n======================== Start Unsupervised Clustering ========================\n")
        start_time = time.time()
        all_feats = np.vstack(list(clip_features.values()))
        all_feats = torch.tensor(all_feats).float()
        if all_feats.shape[0] > 300000:
            _sub = torch.randperm(all_feats.shape[0])[:300000]
            all_feats = all_feats[_sub]
            print(f"[PAPL] subsampled clustering input to 300k (from full set) for speed")
        # Feature normalization
        scaler = StandardScaler()
        all_feats = scaler.fit_transform(all_feats)
        kmeans = K_Means(k=args.k_value, tolerance=1e-4, max_iterations=100, init='k-means++',
                         n_init=5, random_state=args.seed, n_jobs=-1, pairwise_batch_size=512,
                         mode=None)
        kmeans.fit(torch.tensor(all_feats).float())
        print(f"Clustering done, elapsed: {time.time() - start_time:.2f}s, num centers: {kmeans.k}")
        cluster_center_path = os.path.join(args.center_path, f"{args.k_value}_cluster_centers.pth")
        print(f"Cluster centers saved to: {cluster_center_path}")
        centers = kmeans.save_centers(cluster_center_path)

    elif args.PL_type == "text":
        print(f"\n======================== Loading Novel-class Text Embeddings ========================\n")
        text_embeddings = torch.load(args.text_embedding_path)
        centers = text_embeddings.float().cpu()
    else:
        raise ValueError(f"Unsupported Pseudo_Label_type: {args.PL_type}. Only 'prototype' or 'text' are supported.")

    print(f"\n======================== Generating json pseudo-label data ========================\n")
    results_proposals = retrieve_patches_faiss(args.PL_type, clip_features, centers, args.proposals_pkl_path, 512)
    if args.t_sne_gt:
        xml_dir = os.path.join(args.data_root, "Horizontal Bounding Boxes")
        class_names = DIOR_CLASSES[args.num_base_classes:]
        print(f"JQ class_names:{class_names}")
        gt_annotations = generate_gt_annotations_from_xml(xml_dir, class_names) # label=0,1,2,3
    else:
        gt_annotations = None
    json_annotations, sample_feats, sample_pred, sample_gt = build_novel_dict_faiss_json(args, centers, results_proposals, gt_annotations) # label[0-19]

    save_path = os.path.join(args.save_json_path, f"DIOR_vild_cluster{args.k_value}_proposal{args.num_samples}.json")
    with open(save_path, "w") as f:
        json.dump(json_annotations, f)
    print(f"json file saved to: {args.save_json_path}")

    return json_annotations, sample_feats, sample_pred, sample_gt, centers

def retrieve_patches_faiss(PL_type, clip_features, centers, proposals_pkl_path, clip_dim = 512):

    with open(proposals_pkl_path, 'rb') as f:
        proposals = pickle.load(f)
    # Cluster-center processing
    centers_features = np.ascontiguousarray(centers.numpy())
    centers_features = centers_features.reshape(-1, clip_dim)

    ids = [s for s in clip_features]
    all_img_ids = [[] for _ in range(len(centers_features))]
    all_proposals = [[] for _ in range(len(centers_features))]
    all_embeddings = [[] for _ in range(len(centers_features))]
    all_dis = [[] for _ in range(len(centers_features))]
    for i in tqdm(range(0, len(ids) // 1000 + 1)):
        batch_features = []
        batch_proposals = []
        batch_img_ids = []
        for img_id in ids[1000 * i: 1000 * (i + 1)]:
            id = ids.index(img_id)
            try:
                proposal = np.array(proposals[id])
            except IndexError:
                continue
            valid = proposal[:, -1] >= 0.1  # keep proposals with score > 0.1
            features = torch.tensor(clip_features[img_id]).float()
            features = torch.nn.functional.normalize(features, dim=-1, p=2).numpy()
            batch_features.append(features[valid].copy())
            batch_proposals.append(proposal[valid].copy())
            batch_img_ids += [img_id] * valid.sum()
        batch_features = np.concatenate(batch_features)
        batch_proposals = np.concatenate(batch_proposals)

        if PL_type == "prototype":
            index = faiss.IndexFlatL2(clip_dim)
            index.add(batch_features)
            scores, indices = index.search(centers_features, 1000)
        elif PL_type == "text":
            index = faiss.IndexFlatIP(clip_dim)
            index.add(batch_features)
            scores, indices = index.search(centers_features, 1000)
        else:
            raise ValueError(f"Unsupported Pseudo_Label_type: {PL_type}. Only 'prototype' or 'text' are supported.")

        selected_proposals = [batch_proposals[ids] for ids in indices]
        selected_embeddings = [batch_features[ids] for ids in indices]
        selected_img_ids = [[batch_img_ids[k] for k in ids1] for ids1 in indices]
        for j in range(len(centers_features)):
            all_img_ids[j] += selected_img_ids[j]
            all_proposals[j].append(selected_proposals[j])
            all_embeddings[j].append(selected_embeddings[j])
            all_dis[j].append(scores[j])
    for j in range(len(centers_features)):
        all_proposals[j] = np.concatenate(all_proposals[j])
        all_embeddings[j] = np.concatenate(all_embeddings[j])
        all_dis[j] = np.concatenate(all_dis[j])

    saved_dict = {
        "image_ids": all_img_ids,
        "proposals": all_proposals,
        "embeddings": all_embeddings,
        "dis": all_dis,
    }
    return saved_dict

def build_novel_dict_faiss_json(args, centers_features, results_proposals, gt_annotations=None):
    centers_features2 = np.ascontiguousarray(centers_features)
    centers_features2 = centers_features2.reshape(-1, 512)

    saved_dict = {id: [] for id in range(1, len(centers_features2) + 1)}
    annotations = []
    images = []
    image_id_set = set()
    all_features = []
    all_preds = []
    all_gt = []
    for i, id in enumerate(range(1, len(centers_features2) + 1)):
        candidate_img_ids = results_proposals["image_ids"][i]
        candidate_proposals = results_proposals["proposals"][i]
        candidate_embeddings = results_proposals["embeddings"][i]

        candidate_embeddings = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, ord=2, axis=-1, keepdims=True
        )

        if args.PL_type == "prototype":
            index = faiss.IndexFlatL2(512)
            index.add(candidate_embeddings)
            scores, indices = index.search(centers_features2[i: i + 1], args.num_samples)
            scores = scores.reshape([-1])
            indices = indices.reshape([-1])
        elif args.PL_type == "text":
            index = faiss.IndexFlatIP(512)
            index.add(candidate_embeddings)
            scores, indices = index.search(centers_features2[i: i + 1], args.num_samples)
            scores = scores.reshape([-1])
            indices = indices.reshape([-1])
        else:
            raise ValueError(f"Unsupported Pseudo_Label_type: {args.PL_type}. Only 'prototype' or 'text' are supported.")

        selected_proposals = candidate_proposals[indices]
        selected_embedding = candidate_embeddings[indices]
        selected_img_ids = [candidate_img_ids[ids] for ids in indices]
        count = 0

        cluster_labels = []

        for k in range(len(selected_img_ids)):
            if count == args.num_samples:
                break

            if gt_annotations is not None:
                img_id = selected_img_ids[k]
                proposal = selected_proposals[k][:4]

                label = -1

                if img_id in gt_annotations:
                    gt_boxes = gt_annotations[img_id]['boxes']
                    gt_labels = gt_annotations[img_id]['labels']

                    max_iou = 0
                    best_label = None
                    for gt_box, gt_label in zip(gt_boxes, gt_labels):
                        iou = calculate_iou(proposal, gt_box)
                        if iou > max_iou and iou > args.iou_thr: 
                            max_iou = iou
                            best_label = gt_label

                    if best_label is not None:
                        label = best_label
                cluster_labels.append(label)
            else:
                cluster_labels = []
            img_info = {
                "img_id": selected_img_ids[k],
                "proposal": selected_proposals[k],
                "features": selected_embedding[k],
                # "objectness":selected_proposals[k][-1], 
                "dis": scores[k],
            }
            saved_dict[id].append(img_info)
            count += 1

        selected_pred = torch.tensor([i] * args.num_samples)
        all_features.append(torch.tensor(selected_embedding[:args.num_samples]))
        all_preds.append(selected_pred)
        all_gt.append(torch.tensor(cluster_labels[:args.num_samples])) 

    if args.PL_type == "prototype":
        categories = [{"name": f'Unknown Class{id + 1}', "id": id + args.num_base_classes} for id in range(len(centers_features2))]
    elif args.PL_type == "text":
        DIOR_UNSEEN_CLS = list(DIOR_CLASSES[args.num_base_classes:])
        print(f"JQ unseen class_names:{class_names}")
        categories = [{"name": name, "id": id + args.num_base_classes} for id, name in enumerate(DIOR_UNSEEN_CLS)]
    else:
        raise ValueError(f"Unsupported Pseudo_Label_type: {args.PL_type}. Only 'prototype' or 'text' are supported.")

    ann_id = 1
    for id in tqdm(saved_dict.keys()):
        category_id = id - 1
        for ann in saved_dict[id]:
            img_id = ann["img_id"]
            proposal = ann["proposal"]
            left, top, right, bottom = proposal[:4]
            score = proposal[-1]
            dis = ann["dis"]
            if img_id not in image_id_set:
                image_id_set.add(img_id)
                image_path = os.path.join(args.data_root, f"JPEGImages-trainval/{str(img_id) + '.jpg'}")
                pil_image = Image.open(image_path)
                width = pil_image.width
                height = pil_image.height
                images.append(
                    {"file_name": "JPEGImages-trainval/" + str(img_id) + ".jpg", "width": width,
                     "height": height, "id": img_id}
                )
            width_b = right - left
            height_b = bottom - top
            area = float(width_b * height_b)
            ann = {
                "area": area,
                "bbox": [float(left), float(top), float(width_b), float(height_b)],
                "image_id": img_id,
                "id": ann_id,
                "category_id": category_id + args.num_base_classes,
                "iscrowd": 0,
                "score": float(score),
                "dis": float(dis),
            }

            ann_id += 1
            annotations.append(ann)
    json_annotations = {"annotations": annotations, "images": images, "categories": categories}

    return json_annotations, all_features, all_preds, all_gt

def text_cluster_matching(args, centers=None, top_k = 1):
    """
    Analyze the matching between cluster centers and text embeddings.
    """
    print(f"\n======================== Computing matching between cluster centers and text embeddings ========================\n")
    mask = None
    novel_class_names = DIOR_CLASSES[args.num_base_classes:]
    novel_class_dict = dict(enumerate(novel_class_names))
    text_embeddings = torch.load(args.text_embedding_path, map_location="cpu")
    text_embeddings = text_embeddings.float().cpu()
    text_embeddings_norm = torch.nn.functional.normalize(text_embeddings, dim=-1, p=2).numpy()

    centers = centers.reshape(-1, 512).numpy()
    similarity_matrix = centers @ text_embeddings_norm.T
    topk_scores, topk_indices = torch.tensor(similarity_matrix).T.topk(k=top_k, dim=1)


    # Dynamic selection mechanism
    if args.dynamic_k:
        similarity_matrix = torch.tensor(text_embeddings_norm @ centers.T)
        diff = similarity_matrix.max(dim=0)[0] - similarity_matrix.min(dim=0)[0]
        mask = diff >= 2
        mask_id = np.where(mask)[0]
        sub_mat = similarity_matrix[:, mask]
        _, top1_row = sub_mat.max(dim=0)
        topk_indices = torch.tensor([[int(r)] for r in top1_row]).squeeze(1)

        return topk_indices, novel_class_names, mask_id,



    class_results = {}
    current_rank = 0
    if current_rank == 0:
        print(f"\nTop {top_k} cluster centers with highest similarity for each text class:")
    for class_idx, class_name in novel_class_dict.items():
        if class_name not in class_results:
            class_results[class_name] = {}
        # Top-k matching info
        for rank in range(top_k):
            center_id = topk_indices[class_idx][rank].item() + args.num_base_classes
            score = topk_scores[class_idx][rank].item()

            class_results[class_name][f"top{rank}"] = {
                "center": center_id,
                "score": score
            }
        print(f"\nClass '{class_name}' (ID:{class_idx}):")
        for rank in range(top_k):
            match_info = class_results[class_name][f"top{rank}"]
            print(f"  Rank {rank} cluster center: C{match_info['center']} (similarity={match_info['score']:.4f})")

    return topk_indices, novel_class_names, mask


if __name__ == "__main__":
    args = parse_args()

    json_annotations, sample_feats, sample_pred, sample_gt, centers_features = main(args)

    text_topk_indices, novel_class_names, mask_id = text_cluster_matching(args, centers_features, top_k=args.top_p)

    # with open(json_path) as f:
    #     json_annotations = json.load(f)
    sample_feats = torch.stack(sample_feats)
    if args.visualize_json:
        visualize_from_json(json_annotations, args.data_root, output_dir="work_dirs/visualize_outputs")
    if args.t_sne:
        print("\n======================== t-SNE on embeddings, labeled with PL ========================\n")
        sample_pred = torch.stack(sample_pred)
        # text_topk_indices = torch.tensor([14, 9, 17, 11])
        visualize_tsne(sample_feats.reshape(-1,512), sample_pred.reshape(-1), text_topk_indices, novel_class_names, None, mask_id=mask_id)
        print("\n======================== Done ========================\n")
    if args.t_sne_gt:
        print("\n======================== t-SNE on embeddings, labeled with GT ========================\n")
        sample_gt = torch.stack(sample_gt)
        text_topk_indices = torch.tensor([0, 1, 2, 3])
        visualize_tsne(sample_feats.reshape(-1, 512), sample_gt.reshape(-1), text_topk_indices, novel_class_names, args.t_sne_gt)
        print("\n======================== Done ========================\n")