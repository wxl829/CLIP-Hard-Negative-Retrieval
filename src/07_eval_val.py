import argparse

import json
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ========== 1. 基本配置 ==========

parser = argparse.ArgumentParser(description="Evaluate a CLIP checkpoint on the exported validation gallery")
parser.add_argument("--model-path", default="openai/clip-vit-base-patch32")
parser.add_argument("--val-dir", type=Path, default=Path("data/val_gallery"))
parser.add_argument("--output", type=Path, default=Path("outputs/reproduced/eval_pretrained"))
args = parser.parse_args()
batch_size = 16
val_dir = args.val_dir
output_dir = args.output
if (output_dir / "baseline_metrics.json").exists():
    parser.error(f"Evaluation output already exists: {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)

records = json.loads(
    (val_dir / "metadata.json").read_text(encoding="utf-8")
)

n_images = len(records)

# ========== 2. 整理全部文本和真实对应关系 ==========

texts = []
text_to_image = []

for image_idx, record in enumerate(records):
    for caption in record["captions"]:
        texts.append(caption)
        text_to_image.append(image_idx)

n_texts = len(texts)

print("图片数量:", n_images)
print("文本数量:", n_texts)
print("前10条文本对应的图片:", text_to_image[:10])

# ========== 3. 加载预训练 CLIP ==========

model_name = args.model_path
model_kwargs = {"revision": "be35e1cf6ab51bb4daa2b1db53e9f543d4544b47"} if model_name == "openai/clip-vit-base-patch32" else {}

processor = CLIPProcessor.from_pretrained(
    model_name,
    use_fast=False
)

model = CLIPModel.from_pretrained(
    model_name,
    **model_kwargs,
    use_safetensors=True
)

model = model.to(device)
model.eval()

# ========== 4. 分批提取 Embedding ==========

image_features = []
text_features = []

with torch.no_grad():

    for start in range(0, n_images, batch_size):

        end = min(start + batch_size, n_images)
        batch_records = records[start:end]

        batch_images = []
        batch_texts = []

        for record in batch_records:

            image_path = val_dir / "images" / record["filename"]

            with Image.open(image_path) as img:
                batch_images.append(img.convert("RGB"))

            batch_texts.extend(record["captions"])

        inputs = processor(
            images=batch_images,
            text=batch_texts,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        )

        for img in batch_images:
            img.close()

        inputs = inputs.to(device)

        outputs = model(**inputs)

        image_features.append(
            outputs.image_embeds.cpu()
        )

        text_features.append(
            outputs.text_embeds.cpu()
        )

        if end % 160 == 0 or end == n_images:
            print(f"已编码图片: {end}/{n_images}")

# ========== 5. 合并特征 ==========

image_embeds = torch.cat(image_features, dim=0)
text_embeds = torch.cat(text_features, dim=0)

image_embeds = F.normalize(
    image_embeds, p=2, dim=-1
)

text_embeds = F.normalize(
    text_embeds, p=2, dim=-1
)

print("图像特征:", image_embeds.shape)
print("文本特征:", text_embeds.shape)

# ========== 6. 保存特征 ==========

torch.save(
    {
        "image_embeds": image_embeds,
        "text_embeds": text_embeds,
        "text_to_image": text_to_image,
        "texts": texts,
        "filenames": [r["filename"] for r in records]
    },
    output_dir / "val_embeddings.pt"
)

print("验证集 Embedding 保存成功")

# ========== 7. 计算双向相似度 ==========

sim_i2t = image_embeds @ text_embeds.T
sim_t2i = sim_i2t.T

print("I2T:", sim_i2t.shape)
print("T2I:", sim_t2i.shape)

# ========== 8. Recall@K 评测 ==========

gt = torch.tensor(text_to_image, dtype=torch.long)

results = {}

for k in [1, 5, 10]:

    # ----- T2I：文本找图片 -----

    topk_t2i = torch.topk(
        sim_t2i, k=k, dim=1
    ).indices

    # 是否出现在这条文本的Top-K结果中。
    t2i_hits = (topk_t2i == gt.unsqueeze(1)).any(dim=1)


    # ----- I2T：图片找文本 -----

    topk_i2t = torch.topk(
        sim_i2t, k=k, dim=1
    ).indices


    #第一步：将检索出的文本编号转换成所属图片编号
    retrieved_image_ids = gt[topk_i2t]

    # 第二步：生成所有查询图片的编号，并增加一个维度
    query_image_ids = torch.arange(n_images).unsqueeze(1)

    # 第三步：比较是否命中正确图片
    i2t_hits = (retrieved_image_ids == query_image_ids).any(dim=1)


    # ----- 计算平均命中率 -----

    t2i_recall = t2i_hits.float().mean().item() * 100
    i2t_recall = i2t_hits.float().mean().item() * 100

    results[f"T2I_R@{k}"] = round(t2i_recall, 4)
    results[f"I2T_R@{k}"] = round(i2t_recall, 4)

    print(
        f"R@{k} | "
        f"T2I: {t2i_recall:.2f}% | "
        f"I2T: {i2t_recall:.2f}%"
    )

# ========== 9. 保存评测指标 ==========

with open(
    output_dir / "baseline_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\nD3 基线评测完成")
print(results)
