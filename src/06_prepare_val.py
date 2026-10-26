
import json
from pathlib import Path
from datasets import load_dataset

# Export the same validation split used by the training scripts.
dataset = load_dataset("AnyModal/flickr30k", download_mode="reuse_dataset_if_exists")["validation"]

print("验证集图片数量:", len(dataset))

# 2. 创建独立的验证集目录
val_dir = Path("data/val_gallery")
image_dir = val_dir / "images"
image_dir.mkdir(parents=True, exist_ok=True)

records = []

# 3. 保存全部验证集图片和描述
#enumerate() 会在遍历数据集时同时提供索引 i 和当前样本 sample
for i, sample in enumerate(dataset):

    filename = f"{i:04d}.png"
    captions = sample["original_alt_text"]

    # 本次实验要求每张图片恰好5条描述
    if len(captions) != 5:
        raise ValueError(
            f"样本{i}的描述数量为{len(captions)}，预期5条"
        )

    image = sample["image"].convert("RGB")
    image.save(image_dir / filename)

    records.append({
        "filename": filename,
        "original_filename": sample["filename"],
        "captions": captions
    })

# 4. 保存图文对应关系
with open(val_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("保存图片数量:", len(records))
print("第一张图片:", records[0]["filename"])
print("第一张图片描述数量:", len(records[0]["captions"]))
print("Validation数据准备完成！")
