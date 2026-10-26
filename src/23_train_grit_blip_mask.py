#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在现有 GRIT-style 分组训练中，屏蔽 BLIP 高分疑似假负样本。

默认读取：
  outputs/d5/embeddings/metadata.json
  outputs/d7/grit_batches/grouped_batches.json
  outputs/d8/codex_audit_384/blip_vs_codex.csv

BLIP 分数达到阈值的 (candidate image, caption) 关系，仅在它们同时出现在同一个训练 batch
且该 caption 被本 epoch 选中时，才会从双向 CLIP 对比损失中屏蔽。
这些关系不会被改标为正样本，也不会删除图片或文本。

本脚本是 GRIT-style 分组 + BLIP 屏蔽实验，不是 GRIT-VLP 论文的完整复现。
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlickrSubset(Dataset):
    """数据集使用子集相对 image index，以及 metadata 中的全局 caption index。"""

    def __init__(self, hf_train, records, captions, caption_to_image):
        self.records = records
        self.captions = captions
        self.caption_to_image = caption_to_image
        original_indices = [int(rec["index"]) for rec in records]
        if len(set(original_indices)) != len(original_indices):
            raise ValueError("metadata 中存在重复的原始图片索引")
        if not original_indices or min(original_indices) < 0 or max(original_indices) >= len(hf_train):
            raise ValueError("metadata 图片索引越界")
        self.train = hf_train.select(original_indices)
        actual_names = self.train["filename"]
        for i, (actual, record) in enumerate(zip(actual_names, records)):
            if actual != record["filename"]:
                raise ValueError(
                    f"图片索引映射错误：subset image {i}: {actual} != {record['filename']}"
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, pair):
        image_idx, caption_idx = map(int, pair)
        if not 0 <= image_idx < len(self.records):
            raise IndexError(f"图片索引越界：{image_idx}")
        if not 0 <= caption_idx < len(self.captions):
            raise IndexError(f"文本索引越界：{caption_idx}")
        if int(self.caption_to_image[caption_idx]) != image_idx:
            raise ValueError(
                f"caption {caption_idx} 不属于 image {image_idx}"
            )
        row = self.train[image_idx]
        return {
            "image": row["image"].convert("RGB"),
            "caption": self.captions[caption_idx],
            "subset_index": image_idx,
            "caption_index": caption_idx,
        }


def make_collate(processor):
    def collate(rows):
        image_indices = [int(r["subset_index"]) for r in rows]
        caption_indices = [int(r["caption_index"]) for r in rows]
        if len(image_indices) != len(set(image_indices)):
            raise ValueError("同一 batch 出现重复图片，不能直接使用对角线 CLIP 标签")
        if len(caption_indices) != len(set(caption_indices)):
            raise ValueError("同一 batch 出现重复文本，不能直接使用对角线 CLIP 标签")
        encoded = processor(
            images=[r["image"] for r in rows],
            text=[r["caption"] for r in rows],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        return encoded, image_indices, caption_indices

    return collate


class PrecomputedBatchSampler(Sampler):
    """每个 batch 产出 (子集图片索引, 全局 caption 索引)。"""

    def __init__(self, batches, caption_choices):
        self.batches = batches
        self.caption_choices = caption_choices

    def __iter__(self):
        for batch in self.batches:
            yield [(int(i), int(self.caption_choices[i])) for i in batch]

    def __len__(self):
        return len(self.batches)


def load_checked_groups(path, metadata, batch_size, epochs):
    groups = json.loads(path.read_text(encoding="utf-8"))
    records = metadata["records"]
    captions = metadata["captions"]
    owners = metadata["caption_to_image"]
    n = len(records)
    if int(groups["num_images"]) != n:
        raise ValueError("GRIT batch 文件的图片数量与 metadata 不一致")
    if int(groups["batch_size"]) != batch_size:
        raise ValueError("GRIT batch 文件的 batch_size 不一致；请使用对应配置")
    if len(groups["epochs"]) < epochs:
        raise ValueError(
            f"GRIT batch 文件只有 {len(groups['epochs'])} 个 epoch，少于请求的 {epochs} 个"
        )
    if len(owners) != len(captions):
        raise ValueError("metadata 中 captions 与 caption_to_image 长度不一致")

    verified = []
    for e, info in enumerate(groups["epochs"][:epochs], start=1):
        if int(info["epoch"]) != e:
            raise ValueError(f"GRIT batch 文件 epoch 编号错误：预期 {e}")
        choices = info["selected_caption_indices"]
        if len(choices) != n:
            raise ValueError(f"Epoch {e}: caption choices 数量错误")
        for image_idx, text_idx in enumerate(choices):
            if not 0 <= int(text_idx) < len(captions):
                raise ValueError(f"Epoch {e}: caption 索引越界 {text_idx}")
            if int(owners[int(text_idx)]) != image_idx:
                raise ValueError(
                    f"Epoch {e}: caption {text_idx} 不属于 image {image_idx}"
                )
        for mode in ("random", "grit"):
            batches = info[f"{mode}_batches"]
            if len(batches) != math.ceil(n / batch_size):
                raise ValueError(f"Epoch {e}, {mode}: batch 数量错误")
            flat = [int(idx) for batch in batches for idx in batch]
            if len(flat) != n or sorted(flat) != list(range(n)):
                raise ValueError(f"Epoch {e}, {mode}: 图片索引存在遗漏或重复")
            if any(len(batch) != batch_size for batch in batches[:-1]):
                raise ValueError(f"Epoch {e}, {mode}: 非最后一个 batch 大小异常")
            if not 1 <= len(batches[-1]) <= batch_size:
                raise ValueError(f"Epoch {e}, {mode}: 最后一个 batch 大小异常")
        verified.append(info)
    return verified


def load_blip_mask_pairs(path, threshold, metadata):
    """读取 BLIP/Codex 对照表，并严格核对图片、文本索引映射。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 BLIP 审核文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"BLIP 审核文件为空：{path}")

    required = {
        "candidate_index", "text_index", "candidate_image",
        "query_text", "blip_match_score",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"BLIP 审核文件缺少字段：{sorted(missing)}")

    records = metadata["records"]
    captions = metadata["captions"]
    owners = metadata["caption_to_image"]
    selected_rows = 0
    pairs = set()

    for row in rows:
        try:
            score = float(row["blip_match_score"])
            candidate_idx = int(row["candidate_index"])
            text_idx = int(row["text_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BLIP 审核行索引/分数无效：{row}") from exc

        if not math.isfinite(score):
            raise ValueError(f"BLIP 分数不是有限数值：{row}")
        if not 0 <= candidate_idx < len(records):
            raise ValueError(f"candidate_index 越界：{row}")
        if not 0 <= text_idx < len(captions):
            raise ValueError(f"text_index 越界：{row}")
        if row["candidate_image"] != records[candidate_idx]["filename"]:
            raise ValueError(
                "BLIP candidate_image 与 metadata 索引不匹配："
                f"{row['candidate_image']} != {records[candidate_idx]['filename']}"
            )
        if row["query_text"] != captions[text_idx]:
            raise ValueError(f"BLIP query_text 与 text_index 不匹配：{row}")
        owner_idx = int(owners[text_idx])
        if owner_idx == candidate_idx:
            raise ValueError(
                f"审核关系实际是已知正样本，不应作为负样本屏蔽：{row}"
            )

        if score >= threshold:
            selected_rows += 1
            pairs.add((candidate_idx, text_idx))

    if not pairs:
        raise ValueError(
            f"没有 blip_match_score >= {threshold:.3f} 的候选关系，停止训练。"
        )
    return pairs, selected_rows, len(rows)


def scheduled_mask_hits(groups, sampler, mask_pairs, owners):
    """训练前计算各 epoch 中理论上会进入同一 batch 的候选关系数。"""
    counts = []
    for info in groups:
        choices = [int(x) for x in info["selected_caption_indices"]]
        batches = info[f"{sampler}_batches"]
        image_to_batch = {}
        for batch_idx, batch in enumerate(batches):
            for image_idx in batch:
                image_to_batch[int(image_idx)] = batch_idx

        hits = 0
        for candidate_idx, text_idx in mask_pairs:
            owner_idx = int(owners[text_idx])
            if choices[owner_idx] != text_idx:
                continue
            if image_to_batch.get(candidate_idx) == image_to_batch.get(owner_idx):
                hits += 1
        counts.append(hits)
    return counts


def build_batch_mask(image_indices, caption_indices, mask_pairs, device):
    """返回 [B,B] mask：行是图片，列是文本；True 表示从负样本分母中屏蔽。"""
    b = len(image_indices)
    if len(caption_indices) != b:
        raise ValueError("batch 内图片数与文本数不一致")

    rows, cols = [], []
    for i, image_idx in enumerate(image_indices):
        for j, caption_idx in enumerate(caption_indices):
            if (int(image_idx), int(caption_idx)) in mask_pairs:
                if i == j:
                    raise ValueError("屏蔽候选落在对角线正样本位置，拒绝继续训练")
                rows.append(i)
                cols.append(j)

    mask = torch.zeros((b, b), dtype=torch.bool)
    if rows:
        mask[torch.tensor(rows), torch.tensor(cols)] = True
    return mask.to(device)


def as_feature_tensor(x):
    return x if torch.is_tensor(x) else x.pooler_output


@torch.inference_mode()
def evaluate(model, processor, val, device, image_batch=32, text_batch=64):
    """沿用项目已有的验证协议：1014 张图片、每张图片的 5 条真值描述。"""
    model.eval()
    image_features = []
    texts = []
    for start in range(0, len(val), image_batch):
        rows = val[start: start + image_batch]
        images = [im.convert("RGB") for im in rows["image"]]
        for caps in rows["original_alt_text"]:
            if len(caps) != 5:
                raise ValueError("验证集图片不是恰好 5 条描述")
            texts.extend(caps)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
        feats = as_feature_tensor(model.get_image_features(**inputs))
        image_features.append(F.normalize(feats.float(), dim=-1).cpu())

    text_features = []
    for start in range(0, len(texts), text_batch):
        inputs = processor(
            text=texts[start: start + text_batch],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }
        feats = as_feature_tensor(model.get_text_features(**inputs))
        text_features.append(F.normalize(feats.float(), dim=-1).cpu())

    img = torch.cat(image_features, dim=0)
    txt = torch.cat(text_features, dim=0)
    assert len(img) == len(val) and len(txt) == len(val) * 5
    sims = txt @ img.T
    ranked_t2i = sims.topk(10, dim=1).indices
    ranked_i2t = sims.T.topk(10, dim=1).indices
    gt_images = torch.arange(len(txt)) // 5
    gt_text_owners = torch.arange(len(img))
    metrics = {}
    for k in (1, 5, 10):
        t2i_ok = (ranked_t2i[:, :k] == gt_images[:, None]).any(dim=1)
        i2t_ok = ((ranked_i2t[:, :k] // 5) == gt_text_owners[:, None]).any(dim=1)
        metrics[f"T2I_R@{k}"] = round(100 * t2i_ok.float().mean().item(), 4)
        metrics[f"I2T_R@{k}"] = round(100 * i2t_ok.float().mean().item(), 4)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("outputs/d5/embeddings/metadata.json"))
    p.add_argument("--group-file", type=Path, default=Path("outputs/d7/grit_batches/grouped_batches.json"))
    p.add_argument("--audit-file", type=Path, default=Path("outputs/d8/codex_audit_384/blip_vs_codex.csv"))
    p.add_argument("--threshold", type=float, default=0.92)
    p.add_argument("--sampler", choices=("grit", "random"), default="grit")
    p.add_argument("--output", type=Path, default=Path("outputs/d8/ft_grit_blip_mask_t092"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="只检查索引映射与预计屏蔽次数，不加载模型、不训练")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.max_steps < 0 or args.log_every < 1:
        raise ValueError("Require max-steps >= 0 and log-every >= 1")

    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        p.error("epochs >= 1、batch-size >= 2、lr > 0")
    if not 0 <= args.threshold <= 1:
        p.error("threshold 必须在 [0,1]")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    groups = load_checked_groups(args.group_file, metadata, args.batch_size, args.epochs)
    if json.loads(args.group_file.read_text(encoding="utf-8"))["seed"] != args.seed:
        p.error("Group file seed must match --seed for the paired experiment")
    mask_pairs, selected_rows, total_rows = load_blip_mask_pairs(
        args.audit_file, args.threshold, metadata
    )
    owners = metadata["caption_to_image"]
    planned_hits = scheduled_mask_hits(groups, args.sampler, mask_pairs, owners)

    print(f"BLIP 审核记录：{total_rows} 条；分数 >= {args.threshold:.3f}：{selected_rows} 条", flush=True)
    print(f"去重后的候选图文关系：{len(mask_pairs)} 条", flush=True)
    print(f"Sampler={args.sampler}; 每个 epoch 预计触发屏蔽关系数：{planned_hits}", flush=True)
    print(f"全训练计划屏蔽次数：{sum(planned_hits)}", flush=True)
    if sum(planned_hits) == 0:
        raise RuntimeError(
            "这些 BLIP 候选在当前 batch/caption 安排中不会触发屏蔽；"
            "为避免浪费 GPU，本次不启动训练。请检查 GRIT batch 文件与审核样本是否同一子集。"
        )
    if args.dry_run:
        print("Dry-run 通过：索引映射有效，且当前训练计划中存在可触发的屏蔽关系；未加载模型、未训练。")
        return
    if not torch.cuda.is_available():
        p.error("没有可见 CUDA GPU；请先预约 GPU 并设置 CUDA_VISIBLE_DEVICES")

    if args.output.resolve() in {
        Path("outputs/d6/clip_ft_lr1e6_ep10").resolve(),
        Path("outputs/d6/clip_ft_baseline").resolve(),
    }:
        p.error("拒绝覆盖已有 D6 基线目录")
    if (args.output / "history.json").exists():
        p.error(f"输出目录已包含 history.json，为避免覆盖已停止：{args.output}")

    set_seed(args.seed)
    device = torch.device("cuda:0")
    use_amp = not args.no_amp
    print(f"Device={torch.cuda.get_device_name(device)} AMP(fp16)={use_amp}", flush=True)

    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor

    dataset = load_dataset("AnyModal/flickr30k", download_mode="reuse_dataset_if_exists")
    train_data = FlickrSubset(
        dataset["train"], metadata["records"],
        metadata["captions"], metadata["caption_to_image"]
    )
    val_data = None if args.skip_eval else dataset["validation"]

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(
        model_id, revision="be35e1cf6ab51bb4daa2b1db53e9f543d4544b47", use_safetensors=True
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        model_id, use_fast=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    for key in ("metadata", "group_file", "audit_file", "output"):
        settings[key] = str(settings[key])
    settings["group_file_seed"] = json.loads(
        args.group_file.read_text(encoding="utf-8")
    )["seed"]
    settings["blip_mask_unique_pairs"] = len(mask_pairs)
    settings["planned_mask_hits_per_epoch"] = planned_hits
    (args.output / "config.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logs = []
    global_step = 0
    best_score = -math.inf

    for epoch, info in enumerate(groups, start=1):
        batches = info[f"{args.sampler}_batches"]
        choices = info["selected_caption_indices"]
        batch_sampler = PrecomputedBatchSampler(batches, choices)
        loader = DataLoader(
            train_data,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=make_collate(processor),
            pin_memory=True,
        )
        model.train()
        loss_sum = 0.0
        step_count = 0
        epoch_mask_positions = 0
        epoch_masked_batches = 0
        epoch_hit_pairs = set()
        tick = time.time()

        for step, batch_data in enumerate(loader, start=1):
            batch, image_indices, caption_indices = batch_data
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            if epoch == 1 and step == 1:
                print("First batch shapes:", {k: tuple(v.shape) for k, v in batch.items()}, flush=True)
                print("First batch subset indices:", image_indices, flush=True)
                print("First batch caption indices:", caption_indices, flush=True)

            mask_i2t = build_batch_mask(
                image_indices, caption_indices, mask_pairs, device
            )
            masked_positions = int(mask_i2t.sum().item())
            if masked_positions:
                epoch_mask_positions += masked_positions
                epoch_masked_batches += 1
                for i, j in mask_i2t.nonzero(as_tuple=False).cpu().tolist():
                    epoch_hit_pairs.add((int(image_indices[i]), int(caption_indices[j])))

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                labels = torch.arange(batch["pixel_values"].shape[0], device=device)
                logits_i = outputs.logits_per_image.float().masked_fill(mask_i2t, float("-inf"))
                logits_t = outputs.logits_per_text.float().masked_fill(
                    mask_i2t.T, float("-inf")
                )
                loss_i = F.cross_entropy(logits_i, labels)
                loss_t = F.cross_entropy(logits_t, labels)
                loss = (loss_i + loss_t) / 2

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss 非有限值：epoch {epoch}, step {step}, loss={loss.item()}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100.0))

            loss_sum += float(loss.detach().item())
            step_count += 1
            global_step += 1
            if step == 1 or step % args.log_every == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | step {step}/{len(loader)} | "
                    f"global {global_step} | loss={loss.item():.4f} | "
                    f"masked={masked_positions} | sec={time.time()-tick:.1f}",
                    flush=True,
                )
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "mean_train_loss": round(loss_sum / max(1, step_count), 6),
            "blip_threshold": args.threshold,
            "blip_mask_unique_pairs": len(mask_pairs),
            "blip_mask_positions": epoch_mask_positions,
            "blip_masked_batches": epoch_masked_batches,
            "blip_hit_pairs_this_epoch": len(epoch_hit_pairs),
        }
        print(
            f"Epoch {epoch} finished: mean_train_loss={entry['mean_train_loss']}; "
            f"BLIP masked pairs={epoch_mask_positions}, batches={epoch_masked_batches}",
            flush=True,
        )

        if not args.skip_eval:
            metrics = evaluate(model, processor, val_data, device)
            entry.update(metrics)
            print("Validation:", json.dumps(metrics, ensure_ascii=False), flush=True)
            score = metrics["T2I_R@1"] + metrics["I2T_R@1"]
            if score > best_score:
                best_score = score
                best_dir = args.output / "best"
                model.save_pretrained(best_dir, safe_serialization=True)
                processor.save_pretrained(best_dir)
                (best_dir / "metrics.json").write_text(
                    json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"New best checkpoint: {best_dir}", flush=True)

        logs.append(entry)
        (args.output / "history.json").write_text(
            json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    last_dir = args.output / "last"
    model.save_pretrained(last_dir, safe_serialization=True)
    processor.save_pretrained(last_dir)
    print(f"Saved last checkpoint: {last_dir}", flush=True)
    print(f"History: {args.output / 'history.json'}", flush=True)
    print("BLIP 屏蔽只影响训练损失；验证集未参与梯度计算。", flush=True)


if __name__ == "__main__":
    main()
