import json
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from datasets import load_dataset
from transformers import (
    CLIPModel,
    CLIPProcessor
)


# ====================
# 参数
# ====================

device = "cuda" if torch.cuda.is_available() else "cpu"


subset_path = Path(
    "outputs/d5/train_subset/train_subset.json"
)


save_dir = Path(
    "outputs/d5/embeddings"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)


batch_size = 32


# ====================
# 读取子集
# ====================

records = json.loads(
    subset_path.read_text(
        encoding="utf-8"
    )
)


print(
    "图片数量:",
    len(records)
)


# 展开文本

captions = []

caption_to_image = []


for img_idx, r in enumerate(records):

    for cap in r["captions"]:

        captions.append(cap)

        caption_to_image.append(
            img_idx
        )


print(
    "文本数量:",
    len(captions)
)


# ====================
# 加载原始数据
# ====================

dataset = load_dataset(
    "AnyModal/flickr30k",
    download_mode="reuse_dataset_if_exists"
)

train_data = dataset["train"]


# ====================
# CLIP
# ====================

model_name = (
    "openai/clip-vit-base-patch32"
)


processor = CLIPProcessor.from_pretrained(
    model_name,
    use_fast=False
)


model = CLIPModel.from_pretrained(
    model_name,
    revision="be35e1cf6ab51bb4daa2b1db53e9f543d4544b47",
    use_safetensors=True
)


model.to(device)

model.eval()


# ====================
# 图片编码
# ====================

image_features = []


with torch.no_grad():

    for start in range(
        0,
        len(records),
        batch_size
    ):

        batch_records = records[
            start:start+batch_size
        ]


        images = []

        for r in batch_records:

            img = train_data[
                r["index"]
            ]["image"]

            images.append(
                img.convert("RGB")
            )


        inputs = processor(
            images=images,
            return_tensors="pt"
        )


        inputs = {
            k:v.to(device)
            for k,v in inputs.items()
        }


        feats = model.get_image_features(
            **inputs
        )


        feats = F.normalize(
            feats if torch.is_tensor(feats) else feats.pooler_output,
            dim=-1
        )


        image_features.append(
            feats.cpu()
        )


        print(
            f"图片:{min(start+batch_size,len(records))}/{len(records)}"
        )


image_embeds = torch.cat(
    image_features,
    dim=0
)


print(
    "图片Embedding:",
    image_embeds.shape
)


# ====================
# 文本编码
# ====================

text_features = []


with torch.no_grad():

    for start in range(
        0,
        len(captions),
        batch_size
    ):


        batch_text = captions[
            start:start+batch_size
        ]


        inputs = processor(
            text=batch_text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )


        inputs = {
            k:v.to(device)
            for k,v in inputs.items()
        }


        feats = model.get_text_features(
            **inputs
        )


        feats = F.normalize(
            feats if torch.is_tensor(feats) else feats.pooler_output,
            dim=-1
        )


        text_features.append(
            feats.cpu()
        )


        print(
            f"文本:{min(start+batch_size,len(captions))}/{len(captions)}"
        )


text_embeds = torch.cat(
    text_features,
    dim=0
)


print(
    "文本Embedding:",
    text_embeds.shape
)


# ====================
# 保存
# ====================


torch.save(
    image_embeds,
    save_dir / "image_embeds.pt"
)


torch.save(
    text_embeds,
    save_dir / "text_embeds.pt"
)


with open(
    save_dir / "metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "records":records,
            "captions":captions,
            "caption_to_image":caption_to_image
        },
        f,
        ensure_ascii=False
    )


print(
    "Embedding保存完成"
)