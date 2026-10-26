import json
import random
from pathlib import Path

from datasets import load_dataset


# ====================
# 参数
# ====================

NUM_IMAGES = 5000

output_dir = Path(
    "outputs/d5/train_subset"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ====================
# 加载数据
# ====================

dataset = load_dataset(
    "AnyModal/flickr30k"
)

train_data = dataset["train"]


print(
    "训练图片数量:",
    len(train_data)
)


# ====================
# 随机采样图片
# ====================

random.seed(42)


indices = random.sample(
    range(len(train_data)),
    NUM_IMAGES
)


# ====================
# 保存信息
# ====================

records = []


for idx in indices:

    item = train_data[idx]

    records.append(
        {
            "index": idx,
            "filename": item["filename"],
            "img_id": item["img_id"],
            "captions": item["original_alt_text"]
        }
    )


save_path = (
    output_dir /
    "train_subset.json"
)


with open(
    save_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    "保存完成:",
    save_path
)

print(
    "图片数量:",
    len(records)
)

print(
    "文本数量:",
    sum(
        len(r["captions"])
        for r in records
    )
)