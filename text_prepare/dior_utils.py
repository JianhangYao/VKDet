# Author: Yao
# CreatTime: 2024/11/27
# FileName: visdrone
# Description: simple introduction of the code
import argparse
import os
import os.path as osp

import torch
import clip
from tqdm import tqdm


DIOR_OVD_ALL_CLS = [
    'airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
        'golffield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'airport', 'basketballcourt', 'groundtrackfield', 'windmill'
]
DIOR_SEEN_CLS = [
    'airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
        'golffield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle'
]
DIOR_UNSEEN_CLS = [
        'airport', 'basketballcourt', 'groundtrackfield', 'windmill'
]



def article(name):
    return "an" if name[0] in "aeiou" else "a"
    # return "a" or "an" based on the leading vowel


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


class_only = ["{}"]
single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]
# 63 templates


def build_text_embedding_dior(weights_root=None, output_path=None):
    categories = DIOR_SEEN_CLS
    run_on_gpu = torch.cuda.is_available()
    templates = multiple_templates
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_root = weights_root or "weights"
    output_path = output_path or "embeddings/dior/ovd_dior_text_embedding.pth"
    model_name = 'ViT-B/32'
    clip_model, _ = clip.load(model_name, device=device, download_root=weights_root)
    ckpt = torch.load(
        osp.join(weights_root, "RemoteCLIP-ViT-B-32.pt"), map_location=device
    )

    message = clip_model.load_state_dict(ckpt)
    print(message)

    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        # expand templates
        # print(data)
        for category in tqdm(categories):
            texts = [
                template.format(processed_name(category, rm_dot=True), article=article(category))
                for template in templates
            ]
            texts = ["This is " + text if text.startswith("a") or text.startswith("the") else text for text in texts]  # prepend "This is" when needed
            texts = clip.tokenize(texts)  # CLIP context length 77
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        print(all_text_embeddings.shape)
        print(f"Finished, get text embeddings of size {all_text_embeddings.T.size()}")
        print("Saving...")
        os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
        torch.save(all_text_embeddings.T.cpu(), output_path)
        print(f"Saved text embeddings to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DIOR text embeddings")
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    build_text_embedding_dior(args.weights_root, args.output)
