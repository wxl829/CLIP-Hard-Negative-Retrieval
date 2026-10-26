
import json
import torch
from pathlib import Path

# 1. 加载已经保存的特征
data = torch.load(
    "outputs/d3/val_embeddings.pt",
    map_location="cpu",
    weights_only=True
)

records = json.loads(
    Path("data/val_gallery/metadata.json").read_text(
        encoding="utf-8"
    )
)

image_embeds = data["image_embeds"]
text_embeds = data["text_embeds"]

texts = data["texts"]

# 每条文本对应的正确图片编号
gt = torch.tensor(data["text_to_image"])

# 2. 文本检索图片
similarity = text_embeds @ image_embeds.T

# 找出每条文本预测的第一名图片
predictions = similarity.argmax(dim=1)

# 找出第一名预测错误的文本查询
wrong_mask = predictions != gt

wrong_indices = torch.where(wrong_mask)[0]

print("T2I Top-1 错误查询数:", len(wrong_indices))

# 3. 查看前5个错误案例
for query_idx in wrong_indices[:5].tolist():

    predicted_idx = predictions[query_idx].item()
    correct_idx = gt[query_idx].item()

    print("\n--------------------")
    print("查询文本:", texts[query_idx])
    print("正确图片:", records[correct_idx]["original_filename"])
    print("错误检索:", records[predicted_idx]["original_filename"])

    print(
        "正确图片分数:",
        f"{similarity[query_idx, correct_idx].item():.4f}"
    )

    print(
        "错误图片分数:",
        f"{similarity[query_idx, predicted_idx].item():.4f}"
    )


# ========== 4. I2T：图片检索文本错误分析 ==========

# 1. 计算图片与文本的相似度
sim_i2t = image_embeds @ text_embeds.T

# 2. 找到每张图片检索出的第一名文本
top1_text_indices = sim_i2t.argmax(dim=1)

# Shape: [1014]
print("Top-1文本索引Shape:", top1_text_indices.shape)

# 3. 查询第一名文本实际属于哪张图片
predicted_image_ids = gt[top1_text_indices]

# 4. 生成当前所有查询图片的编号
query_image_ids = torch.arange(len(records))

# 5. 找出预测错误的查询图片
wrong_mask_i2t = predicted_image_ids != query_image_ids

wrong_indices_i2t = torch.where(wrong_mask_i2t)[0]

print("\nI2T Top-1错误查询数:", len(wrong_indices_i2t))

# 6. 查看前5个错误案例
for image_idx in wrong_indices_i2t[:5].tolist():

    predicted_text_idx = top1_text_indices[image_idx].item()

    print("\n============================")
    print("查询图片:", records[image_idx]["original_filename"])

    print("\n模型检索出的第一名文本:")
    print(texts[predicted_text_idx])

    print("\n这张图片的5条正确描述:")
    for i, caption in enumerate(records[image_idx]["captions"], start=1):
        print(f"{i}. {caption}")

    # 模型错误检索的分数
    wrong_score = sim_i2t[
        image_idx, predicted_text_idx
    ].item()

    # 五条正确描述中分数最高的一条
    correct_scores = sim_i2t[image_idx, gt == image_idx]
    best_correct_score = correct_scores.max().item()

    print(f"\n错误Top-1分数: {wrong_score:.4f}")
    print(f"最佳正确描述分数: {best_correct_score:.4f}")
