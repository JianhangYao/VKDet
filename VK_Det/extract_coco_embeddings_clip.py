import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

import argparse
import clip
import time
from mmdet.datasets import CocoDataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torch import nn



import os

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
os.environ['MASTER_ADDR'] = 'localhost'



os.environ['MASTER_PORT'] = '29501'



# dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
# Init distributed environment
dist.init_process_group(backend='nccl', rank = 0, world_size = 1)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class CocoCroppedProposals(CocoDataset):
    n_px = 224  # resize to 224x224

    clip_transform = Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),# resize, center-crop, normalize
        ]
    )
    scales = [1.0, 1.5]

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file="proposals/coco/train_coco_proposals.pkl",
        test_mode=False,
        filter_empty_gt=False,
        proposal_id_map=None,
    ):
        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
            proposal_id_map,
        )
        self.num_proposals_per_img = list(len(proposal) for proposal in self.proposals)

        self.num_proposals_per_img.insert(0, 0)
        self.num_proposals_per_img_cum = np.cumsum(self.num_proposals_per_img)# cumulative sum of proposals per image

    def __len__(self):
        return self.num_proposals_per_img_cum[-1]

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        img_id = np.searchsorted(self.num_proposals_per_img_cum, idx, side="right") - 1# map flat index to image id
        proposal_id = idx - self.num_proposals_per_img_cum[img_id]

        img_info = self.data_infos[img_id]
        if self.img_prefix is not None:
            filename = os.path.join(self.img_prefix, img_info["filename"])
        else:
            filename = img_info["filename"]

        image = Image.open(filename)
        proposal = self.proposals[img_id][proposal_id][:4]# box coords (first 4 elements)
        bboxes, bboxes15 = self.get_bboxes_and_bboxes15(proposal, img_info["height"], img_info["width"])# receptive-field boxes
        img, img15 = image.crop(bboxes), image.crop(bboxes15)
        img = self.clip_transform(img)
        img15 = self.clip_transform(img15)







        return img, img15, idx

    def get_bboxes_and_bboxes15(self, bboxes, h, w):
        def box_coord_2int(box):
            box = (
                np.floor(box[0] - 0.001),
                np.floor(box[1] - 0.001),
                np.ceil(box[2] + 0.001),
                np.ceil(box[3] + 0.001),
            )
            return (
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], w),
                min(box[3], h),
            )  # clip boxes to image boundary

        assert len(bboxes) == 4
        bboxes15 = (
            1.25 * bboxes[0] - 0.25 * bboxes[2],
            1.25 * bboxes[1] - 0.25 * bboxes[3],
            1.25 * bboxes[2] - 0.25 * bboxes[0],
            1.25 * bboxes[3] - 0.25 * bboxes[1],
        )
        bboxes = box_coord_2int(bboxes)
        bboxes15 = box_coord_2int(bboxes15)
        return bboxes, bboxes15   # receptive-field boxes

    def _set_group_flag(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Extract COCO embeddings")
    parser.add_argument("--data_root", default="data/coco/", help="data root")
    parser.add_argument("--split", default="train", help="data split")
    parser.add_argument("--proposal_file", default="proposals/coco/train_coco_proposals.pkl", help="path to pre-computed proposals")
    parser.add_argument("--clip_root", default="weights", help="clip model path")
    parser.add_argument("--num_workers", default=16, type=int, help="num workers per gpu") # 48
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per gpu")
    parser.add_argument("--save_path1", default="clip_img_embeddings.pth", help="path to save output")
    parser.add_argument("--save_path2", default="patch_embeddings.pth", help="path to save output")
    parser.add_argument("--save_path3", default="attn_weights.pth", help="path to save output")
    parser.add_argument("--local_rank",default=1, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # distributed settings
    rank = dist.get_rank()
    # rank = 0
    world_size = dist.get_world_size()
    # world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device_id = int(os.environ["LOCAL_RANK"])
    device_id = int(0)
    torch.cuda.set_device(device_id)

    data_root = args.data_root
    dataset = CocoCroppedProposals(
        pipeline=[],
        ann_file=os.path.join(data_root, f"annotations/instances_{args.split}2017.json"),
        proposal_file=args.proposal_file,
        img_prefix=data_root + f"{args.split}2017",
    )
    # dataset = dataset.cuda()
    clip_model, _ = clip.load("ViT-B/32", device=device, download_root=args.clip_root)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad_(False)  # freeze parameters

    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler,pin_memory=True)

    embeddings = torch.HalfTensor(
        torch.HalfStorage.from_file(args.save_path1, shared=True, size=len(dataset) * clip_model.visual.output_dim)
    ).reshape(len(dataset), -1)
    patch_embeddings = torch.HalfTensor(
        torch.HalfStorage.from_file(args.save_path2, shared=True, size=len(dataset) * 49 * clip_model.visual.output_dim)
    ).reshape(len(dataset), 7, 7, -1)
    attention_weights = torch.HalfTensor(
        torch.HalfStorage.from_file(args.save_path2, shared=True, size=len(dataset) * 12 * 50 * 50)
    ).reshape(len(dataset), 12, 50, 50)
    start = time.time()
    with torch.no_grad():
        for step, (img, img15, idx) in enumerate(dataloader):
            if rank == 0:
                if step % 1000 == 0:
                    print(f"Step: {step}/{len(dataloader)}, Time: {(time.time() - start):.2f}s")
            img = img.to(device)
            img15 = img15.to(device)

            # patches_img = patches_img.to(device)
            clip_image_features, patch_embed, attn_weights= clip_model.encode_image(img)
            clip_image_features15, patch_embed15, attn_weights15= clip_model.encode_image(img15)

            # width = 768
            # print(clip_image_features.shape)
            patch_embed = patch_embed.to(device) # NOTE: patch_embed would normally go through the iBOT head for final embeddings
            patch_embed15 = patch_embed15.to(device)
            if isinstance(attn_weights, list):
                attn_weights = torch.stack(attn_weights)

            attn_weights = attn_weights.to(device)
            attn_weights = attn_weights.permute(1, 0, 2, 3)
            # print(attn_weights.shape)
            # attn_weights = attn_weights.to(device)
            # ln_post = LayerNorm(width)
            # patch_embed = ln_post(patch_embed).to(device)
            # patch_embed15 = ln_post(patch_embed15).to(device)




            # width = 768
            # output_dim = 512
            # scale = width ** -0.5
            # projection_for_patch = nn.Linear(width, output_dim)
            # nn.init.xavier_uniform_(projection_for_patch.weight)
            # nn.init.constant_(projection_for_patch.bias, 0)
            # projection_for_patch = nn.Parameter(scale * torch.randn(width, output_dim))
            # patch_embed_reshaped = patch_embed.view(-1, 768)
            # patch_embed2 = patch_embed_reshaped @ projection_for_patch
            # patch_embed2 = patch_embed2.view(32, 49, output_dim)
            # print(patch_embed2.shape)
            # # patch_embed215 = projection_for_patch(patch_embed15)
            # print(patch_embed2.shape)
            # print(patch_embed215.shape)
            # clip_patches_features = []
            # for i, patch_img in enumerate(patches_img):
            #     patch_img = patch_img.to(device)
            #     clip_patch_features = clip_model.encode_image(patch_img).to(device)
            #     clip_patches_features.append(clip_patch_features).to(device)
            # print(clip_patches_features.shape())
            patch_embed = F.normalize(patch_embed, p=2, dim=1)
            patch_embed15 = F.normalize(patch_embed15, p=2, dim=1)


            clip_image_features_single = clip_image_features + clip_image_features15
            patch_embed_single = patch_embed + patch_embed15
            # print(patch_embed_single.shape)
            clip_image_features = F.normalize(clip_image_features_single, p=2, dim=1)

            # attn_weights = F.normalize(attn_weights, p=2, dim=1)
            embeddings[idx] = clip_image_features.cpu()
            patch_embeddings[idx] = patch_embed_single.cpu()
            attention_weights[idx] = attn_weights.cpu()



if __name__ == "__main__":
    main()
