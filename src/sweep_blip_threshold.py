import argparse
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="outputs/d8/codex_audit_384/blip_vs_codex.csv"
)
parser.add_argument(
    "--output",
    default="outputs/d8/codex_audit_384/blip_threshold_sweep.csv"
)
args = parser.parse_args()

df = pd.read_csv(args.input)
df.columns = df.columns.str.strip()

print("输入文件：", args.input)
print("字段：", df.columns.tolist())

# 自动识别匹配分数列
score_candidates = [
    "match_score",
    "blip_match_score",
    "itm_score",
    "score",
]
score_col = next(
    (c for c in score_candidates if c in df.columns),
    None,
)

# 自动识别 Codex 标签列
label_candidates = ["label", "codex_label"]
label_col = next(
    (c for c in label_candidates if c in df.columns),
    None,
)

if score_col is None or label_col is None:
    raise ValueError(
        "未能自动识别分数列或标签列。"
        f"当前字段：{df.columns.tolist()}"
    )

df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df[label_col] = df[label_col].astype(str).str.strip().str.lower()

# 只评价两类明确标签，排除 uncertain
df = df[
    df[label_col].isin([
        "possible_false_negative",
        "valid_negative",
    ])
].dropna(subset=[score_col]).copy()

y = (
    df[label_col] == "possible_false_negative"
).astype(int).to_numpy()

scores = df[score_col].to_numpy()

print(f"\n参与评价：{len(df)} 条")
print(f"潜在假负样本：{int(y.sum())} 条")
print(f"有效负样本：{int((y == 0).sum())} 条")
print(f"使用分数列：{score_col}")
print(f"使用标签列：{label_col}")

rows = []

for threshold in np.round(np.arange(0, 1.001, 0.01), 2):
    pred = scores >= threshold

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    specificity = tn / (tn + fp) if tn + fp else 0.0

    rows.append({
        "threshold": threshold,
        "predicted_positive": tp + fp,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "specificity": specificity,
    })

result = pd.DataFrame(rows)

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output, index=False, float_format="%.6f")

def show(title, data):
    print(f"\n========== {title} ==========")
    cols = [
        "threshold", "predicted_positive",
        "TP", "FP", "TN", "FN",
        "precision", "recall", "F1",
    ]
    print(data[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 对照原先使用的0.5
show(
    "阈值0.50",
    result[np.isclose(result["threshold"], 0.50)]
)

# 按F1排序
show(
    "F1最高的前10个阈值（探索性结果）",
    result.sort_values("F1", ascending=False).head(10)
)

# 查找Precision至少0.70、Recall至少0.30的候选
candidates = result[
    (result["precision"] >= 0.70)
    & (result["recall"] >= 0.30)
    & (result["predicted_positive"] > 0)
]

if len(candidates):
    show(
        "Precision>=0.70 且 Recall>=0.30 的候选",
        candidates.sort_values("F1", ascending=False).head(10)
    )
else:
    print(
        "\n没有阈值同时满足 Precision>=0.70、"
        "Recall>=0.30。不要强行选阈值，应检查分数与标签的区分能力。"
    )

print("\n完整扫描结果已保存：", output)
print("注意：所有指标衡量的是与Codex标签的一致性，不是人工真值准确率。")
