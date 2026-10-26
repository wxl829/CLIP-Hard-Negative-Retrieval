#!/usr/bin/env python3
"""Fair CLIP fine-tuning: random vs STATIC GRIT-inspired Top10 batches.

Requires (created by src/20_build_grit_batches.py --epochs 10 --batch-size 16 --seed 42):
    outputs/d7/grit_batches/grouped_batches.json
    outputs/d5/embeddings/metadata.json

The two arms use *exactly the same known-positive caption for each image/epoch*.
Only the grouping/order of 5000 images differs. Both start from the same CLIP weights.
This is static GRIT-STYLE grouping, NOT a reproduction of GRIT-VLP.

Run from project root:
    CUDA_VISIBLE_DEVICES=2 python src/21_train_clip_grouped.py \
        --sampler random --epochs 10 --lr 1e-6 \
        --output outputs/d7/ft_random_matched
    CUDA_VISIBLE_DEVICES=2 python src/21_train_clip_grouped.py \
        --sampler grit --epochs 10 --lr 1e-6 \
        --output outputs/d7/ft_grit_style
"""
import argparse
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
    """Dataset indices are SUBSET-relative; each item is (image_idx, caption_idx)."""

    def __init__(self, hf_train, records, captions, caption_to_image):
        self.records = records
        self.captions = captions
        self.caption_to_image = caption_to_image
        original_indices = [int(rec["index"]) for rec in records]
        if len(set(original_indices)) != len(original_indices):
            raise ValueError("Duplicate original train indices")
        if not original_indices or min(original_indices) < 0 or max(original_indices) >= len(hf_train):
            raise ValueError("Original train index out of range")
        self.train = hf_train.select(original_indices)
        # Check entire filename mapping without decoding 5000 image files.
        actual_names = self.train["filename"]
        for i, (actual, record) in enumerate(zip(actual_names, records)):
            if actual != record["filename"]:
                raise ValueError(f"Image index mismatch at subset image {i}: {actual} != {record['filename']}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, pair):
        # BatchSampler sends a pair of indices instead of an ordinary integer.
        image_idx, caption_idx = pair
        image_idx, caption_idx = int(image_idx), int(caption_idx)
        if not (0 <= image_idx < len(self.records)):
            raise IndexError(f"Image idx out of range: {image_idx}")
        if not (0 <= caption_idx < len(self.captions)):
            raise IndexError(f"Caption idx out of range: {caption_idx}")
        if int(self.caption_to_image[caption_idx]) != image_idx:
            raise ValueError(f"Caption {caption_idx} does not belong to image {image_idx}")
        row = self.train[image_idx]
        return {
            "image": row["image"].convert("RGB"),
            "caption": self.captions[caption_idx],
            "subset_index": image_idx,
        }


def make_collate(processor):
    def collate(rows):
        ids = [r["subset_index"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate image in batch: unsafe with diagonal CLIP contrastive loss")
        return processor(
            images=[r["image"] for r in rows],
            text=[r["caption"] for r in rows],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

    return collate


class PrecomputedBatchSampler(Sampler):
    """Yield lists of (subset_image_idx, global_caption_idx) for one epoch."""

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
        raise ValueError("Group file image count mismatch with metadata")
    if int(groups["batch_size"]) != batch_size:
        raise ValueError("Group file batch size mismatch; regenerate groups with requested batch size")
    if len(groups["epochs"]) < epochs:
        raise ValueError(f"Group file has {len(groups['epochs'])} epochs, requested {epochs}; regenerate groups")
    if len(owners) != len(captions):
        raise ValueError("Invalid metadata: captions and caption_to_image lengths disagree")
    verified = []
    for e, info in enumerate(groups["epochs"][:epochs], start=1):
        if int(info["epoch"]) != e:
            raise ValueError(f"Wrong epoch index at {e}")
        choices = info["selected_caption_indices"]
        if len(choices) != n:
            raise ValueError(f"Epoch {e}: wrong caption choices length")
        for image_idx, text_idx in enumerate(choices):
            if not (0 <= text_idx < len(captions)) or int(owners[text_idx]) != image_idx:
                raise ValueError(f"Epoch {e}: caption {text_idx} not from image {image_idx}")
        for mode in ("random", "grit"):
            batches = info[f"{mode}_batches"]
            if len(batches) != math.ceil(n / batch_size):
                raise ValueError(f"Epoch {e}, {mode}: bad number of batches")
            flat = [int(idx) for batch in batches for idx in batch]
            if len(flat) != n or sorted(flat) != list(range(n)):
                raise ValueError(f"Epoch {e}, {mode}: duplicate/missing subset image indices")
            if any(len(batch) != batch_size for batch in batches[:-1]):
                raise ValueError(f"Epoch {e}, {mode}: nonfinal short batch")
            if not (1 <= len(batches[-1]) <= batch_size):
                raise ValueError(f"Epoch {e}, {mode}: invalid last batch")
        verified.append(info)
    return verified


def as_feature_tensor(x):
    return x if torch.is_tensor(x) else x.pooler_output


@torch.inference_mode()
def evaluate(model, processor, val, device, image_batch=32, text_batch=64):
    """Same validation protocol as src/19_train_clip_baseline.py."""
    model.eval()
    image_features = []
    texts = []
    for start in range(0, len(val), image_batch):
        rows = val[start: start + image_batch]
        images = [im.convert("RGB") for im in rows["image"]]
        for caps in rows["original_alt_text"]:
            if len(caps) != 5:
                raise ValueError("Validation image does not have five captions")
            texts.extend(caps)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
        feats = as_feature_tensor(model.get_image_features(**inputs))
        image_features.append(F.normalize(feats.float(), dim=-1).cpu())

    text_features = []
    for start in range(0, len(texts), text_batch):
        inputs = processor(text=texts[start: start + text_batch], padding="max_length",
                           max_length=77, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
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
    p.add_argument("--sampler", choices=("random", "grit"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.max_steps < 0 or args.log_every < 1:
        raise ValueError("Require max-steps >= 0 and log-every >= 1")
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        p.error("epochs >= 1, batch-size >= 2, lr > 0 required")
    if not torch.cuda.is_available():
        p.error("No CUDA device visible; book your GPU and set CUDA_VISIBLE_DEVICES")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    groups = load_checked_groups(args.group_file, metadata, args.batch_size, args.epochs)
    if json.loads(args.group_file.read_text(encoding="utf-8"))["seed"] != args.seed:
        p.error("Group file seed must match --seed for the paired experiment")
    # No accidental output overwrite of baseline or other arm.
    if args.output.resolve() in {Path("outputs/d6/clip_ft_lr1e6_ep10").resolve(),
                                 Path("outputs/d6/clip_ft_baseline").resolve()}:
        p.error("Refusing to overwrite existing D6 baseline directory")
    if (args.output / "history.json").exists():
        p.error(f"Output directory contains previous training history: {args.output}")

    set_seed(args.seed)
    device = torch.device("cuda:0")  # mapped GPU after CUDA_VISIBLE_DEVICES
    use_amp = not args.no_amp
    print(f"Device={torch.cuda.get_device_name(device)} AMP(fp16)={use_amp}", flush=True)
    print(f"Sampler={args.sampler}; epochs={args.epochs}; batch_size={args.batch_size}", flush=True)

    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor

    dataset = load_dataset("AnyModal/flickr30k", download_mode="reuse_dataset_if_exists")
    train_data = FlickrSubset(dataset["train"], metadata["records"],
                              metadata["captions"], metadata["caption_to_image"])
    val_data = None if args.skip_eval else dataset["validation"]
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id, revision="be35e1cf6ab51bb4daa2b1db53e9f543d4544b47",
                                      use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    for key in ("metadata", "group_file", "output"):
        settings[key] = str(settings[key])
    settings["group_file_seed"] = json.loads(args.group_file.read_text(encoding="utf-8"))["seed"]
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    logs = []
    global_step = 0
    best_score = -math.inf
    for epoch, info in enumerate(groups, start=1):
        precomputed_batches = info[f"{args.sampler}_batches"]
        choices = info["selected_caption_indices"]
        batch_sampler = PrecomputedBatchSampler(precomputed_batches, choices)
        loader = DataLoader(train_data, batch_sampler=batch_sampler,
                            num_workers=0, collate_fn=make_collate(processor), pin_memory=True)
        model.train()
        loss_sum, step_count = 0.0, 0
        tick = time.time()
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if epoch == 1 and step == 1:
                print("First batch shapes:", {k: tuple(v.shape) for k, v in batch.items()}, flush=True)
                print("First batch subset indices:", precomputed_batches[0], flush=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                labels = torch.arange(batch["pixel_values"].shape[0], device=device)
                loss_i = F.cross_entropy(outputs.logits_per_image.float(), labels)
                loss_t = F.cross_entropy(outputs.logits_per_text.float(), labels)
                loss = (loss_i + loss_t) / 2
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss epoch {epoch}, step {step}: {loss.item()}")
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
                print(f"Epoch {epoch}/{args.epochs} | step {step}/{len(loader)} | global {global_step} | "
                      f"loss={loss.item():.4f} | sec={time.time()-tick:.1f}", flush=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        entry = {"epoch": epoch, "global_step": global_step,
                 "mean_train_loss": round(loss_sum / max(1, step_count), 6)}
        print(f"Epoch {epoch} finished: mean_train_loss={entry['mean_train_loss']}", flush=True)
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
                (best_dir / "metrics.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
                print(f"New best checkpoint: {best_dir}", flush=True)
        logs.append(entry)
        (args.output / "history.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    last_dir = args.output / "last"
    model.save_pretrained(last_dir, safe_serialization=True)
    processor.save_pretrained(last_dir)
    print(f"Saved last checkpoint: {last_dir}", flush=True)
    print(f"History: {args.output / 'history.json'}", flush=True)
    print("Validation/test images have not contributed gradients.", flush=True)


if __name__ == "__main__":
    main()
