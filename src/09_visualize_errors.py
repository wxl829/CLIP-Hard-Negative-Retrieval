
import json
import textwrap
import torch

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ========== 1. 读取已有数据 ==========

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

gt = torch.tensor(data["text_to_image"])

sim_t2i = text_embeds @ image_embeds.T
sim_i2t = sim_t2i.T

output_dir = Path("outputs/d3/error_vis")
output_dir.mkdir(parents=True, exist_ok=True)

image_dir = Path("data/val_gallery/images")

# ========== 2. 可视化辅助函数 ==========

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 23)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 29)
except OSError:
    font = ImageFont.load_default()
    title_font = font


def draw_wrapped(draw, text, y):
    """自动换行绘制英文文本"""
    for line in textwrap.wrap(str(text), width=95):
        draw.text((40, y), line, fill="black", font=font)
        y += 35
    return y


def draw_photo(canvas, image_idx, x):
    """把指定图片放进报告"""
    path = image_dir / records[image_idx]["filename"]

    with Image.open(path) as image:
        image = ImageOps.contain(
            image.convert("RGB"),
            (670, 460)
        )

        px = x + (670 - image.width) // 2
        py = 220 + (460 - image.height) // 2

        canvas.paste(image, (px, py))


def save_report(name, heading, left_idx, right_idx,
                left_label, right_label, lines):

    canvas = Image.new("RGB", (1480, 1320), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (40, 25), heading,
        fill="black", font=title_font
    )

    draw.text((40, 175), left_label, fill="black", font=font)
    draw.text((760, 175), right_label, fill="black", font=font)

    draw_photo(canvas, left_idx, 40)
    draw_photo(canvas, right_idx, 760)

    y = 720

    for line in lines:
        y = draw_wrapped(draw, line, y)
        y += 12

    canvas.save(output_dir / name)


# ========== 3. T2I：文本检索图片 ==========

predicted_images = sim_t2i.argmax(dim=1)
wrong_t2i = torch.where(predicted_images != gt)[0]

# 避免连续选择同一张正确图片的不同描述
seen_images = set()
count = 0

for query_idx in wrong_t2i.tolist():

    correct_idx = gt[query_idx].item()

    if correct_idx in seen_images:
        continue

    seen_images.add(correct_idx)

    predicted_idx = predicted_images[query_idx].item()

    correct_score = sim_t2i[query_idx, correct_idx].item()
    wrong_score = sim_t2i[query_idx, predicted_idx].item()

    count += 1

    save_report(
        name=f"t2i_{count:02d}.png",
        heading=f"T2I annotated mismatch | Query {query_idx}",
        left_idx=correct_idx,
        right_idx=predicted_idx,
        left_label="Ground-truth image",
        right_label="Model Top-1 image",
        lines=[
            f"QUERY: {texts[query_idx]}",
            f"Correct image: {records[correct_idx]['original_filename']}",
            f"Retrieved image: {records[predicted_idx]['original_filename']}",
            f"Correct score: {correct_score:.4f}",
            f"Top-1 score: {wrong_score:.4f}",
            f"Score gap: {wrong_score - correct_score:.4f}"
        ]
    )

    if count == 5:
        break


# ========== 4. I2T：图片检索文本 ==========

predicted_texts = sim_i2t.argmax(dim=1)

# 每条预测文本原本对应的图片
predicted_owners = gt[predicted_texts]

query_image_ids = torch.arange(len(records))

wrong_i2t = torch.where(
    predicted_owners != query_image_ids
)[0]

for count, image_idx in enumerate(wrong_i2t[:5].tolist(), 1):

    text_idx = predicted_texts[image_idx].item()
    source_image_idx = gt[text_idx].item()

    wrong_score = sim_i2t[image_idx, text_idx].item()
    best_correct = sim_i2t[image_idx, gt == image_idx].max().item()

    lines = [
        f"MODEL TOP-1 TEXT: {texts[text_idx]}",
        f"Top-1 score: {wrong_score:.4f}",
        f"Best ground-truth score: {best_correct:.4f}",
        "FIVE GROUND-TRUTH CAPTIONS:"
    ]

    for j, caption in enumerate(records[image_idx]["captions"], 1):
        lines.append(f"{j}. {caption}")

    save_report(
        name=f"i2t_{count:02d}.png",
        heading=f"I2T annotated mismatch | Image {image_idx}",
        left_idx=image_idx,
        right_idx=source_image_idx,
        left_label="Query image",
        right_label="Image paired with retrieved text",
        lines=lines
    )

print("可视化报告生成完成！")
print("T2I报告: 5张")
print("I2T报告: 5张")
print("输出目录:", output_dir)
