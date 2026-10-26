"""Fine-tune CLIP ViT-B/32 on the fixed 5k Flickr30k train subset.

One image + ONE randomly selected ground-truth caption per step; each image
appears only once in each epoch. The validation split is NEVER used to train.

Smoke test (one free V100, GPU index example):
    CUDA_VISIBLE_DEVICES=1 python src/19_train_clip_baseline.py \
        --max-steps 10 --skip-eval --output outputs/d6/clip_ft_smoke

Full baseline:
    CUDA_VISIBLE_DEVICES=1 python src/19_train_clip_baseline.py \
        --epochs 2 --batch-size 16 --output outputs/d6/clip_ft_baseline

The pre-trained weights are loaded with safetensors as in the existing project.
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
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlickrSubset(Dataset):
    """Keep index mapping exactly as in outputs/d5/embeddings/metadata.json."""

    def __init__(self, hf_train, records):
        self.records = records
        original_indices = [int(rec["index"]) for rec in records]
        if len(set(original_indices)) != len(original_indices):
            raise ValueError("Duplicate original train indices in subset records")
        if min(original_indices) < 0 or max(original_indices) >= len(hf_train):
            raise ValueError("Original train index out of range")
        self.train = hf_train.select(original_indices)

        # Catch the prior confusion between subset-relative and original indices.
        for i in range(min(5, len(records))):
            actual = self.train[i]["filename"]
            expected = records[i]["filename"]
            if actual != expected:
                raise ValueError(f"Data alignment error at subset image {i}: {actual} != {expected}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, subset_idx):
        row = self.train[subset_idx]
        rec = self.records[subset_idx]
        image = row["image"].convert("RGB")
        # Random selection from five TRUE captions, never Top-10 negatives.
        caption = random.choice(rec["captions"])
        return {"image": image, "caption": caption, "subset_index": subset_idx}


def make_collate(processor):
    def collate(rows):
        subset_indices = [r["subset_index"] for r in rows]
        if len(subset_indices) != len(set(subset_indices)):
            raise ValueError("Duplicate image in batch; ordinary diagonal CLIP loss is unsafe")
        inputs = processor(
            images=[r["image"] for r in rows],
            text=[r["caption"] for r in rows],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        return inputs

    return collate


def as_feature_tensor(feature_output):
    """Transformers versions differ in get_*_features return type."""
    if torch.is_tensor(feature_output):
        return feature_output
    return feature_output.pooler_output


@torch.inference_mode()
def evaluate(model, processor, val, device, image_batch=32, text_batch=64):
    """Recall@K with 1,014 images and all five GT captions per val image."""
    model.eval()
    image_features = []
    texts = []

    for start in range(0, len(val), image_batch):
        rows = val[start : start + image_batch]
        images = [im.convert("RGB") for im in rows["image"]]
        for captions in rows["original_alt_text"]:
            if len(captions) != 5:
                raise ValueError("Validation image does not have exactly 5 captions")
            texts.extend(captions)
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
        feats = as_feature_tensor(model.get_image_features(**inputs))
        image_features.append(F.normalize(feats.float(), dim=-1).cpu())

    text_features = []
    for start in range(0, len(texts), text_batch):
        inputs = processor(
            text=texts[start : start + text_batch],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        feats = as_feature_tensor(model.get_text_features(**inputs))
        text_features.append(F.normalize(feats.float(), dim=-1).cpu())

    images = torch.cat(image_features, dim=0)  # [1014, 512]
    captions = torch.cat(text_features, dim=0)  # [5070, 512]
    assert len(images) == len(val) and len(captions) == len(val) * 5

    # T2I: each text has exactly one annotated original image.
    t2i = captions @ images.T                   # [5070, 1014]
    t2i_ranked = t2i.topk(10, dim=1).indices    # [5070, 10]
    gt_images = torch.arange(len(captions)) // 5

    # I2T: any of the five annotated texts for this image is correct.
    i2t = t2i.T                                 # [1014, 5070]
    i2t_ranked = i2t.topk(10, dim=1).indices    # [1014, 10]
    gt_i2t_images = torch.arange(len(images))

    metrics = {}
    for k in (1, 5, 10):
        t2i_ok = (t2i_ranked[:, :k] == gt_images[:, None]).any(dim=1)
        i2t_ok = ((i2t_ranked[:, :k] // 5) == gt_i2t_images[:, None]).any(dim=1)
        metrics[f"T2I_R@{k}"] = round(100 * t2i_ok.float().mean().item(), 4)
        metrics[f"I2T_R@{k}"] = round(100 * i2t_ok.float().mean().item(), 4)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("outputs/d5/embeddings/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/d6/clip_ft_baseline"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Global optimization step limit; 0 means full epochs")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 0 or args.log_every < 1:
        raise ValueError("Require max-steps >= 0 and log-every >= 1")

    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        raise ValueError("Require epochs>=1, batch-size>=2, lr>0")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible. Reserve a GPU and set CUDA_VISIBLE_DEVICES first.")
    if (args.output / "history.json").exists():
        parser.error(f"Output already contains training history: {args.output}")
    set_seed(args.seed)
    device = torch.device("cuda:0")  # logical 0 after CUDA_VISIBLE_DEVICES filtering
    use_amp = not args.no_amp
    print(f"Device: {torch.cuda.get_device_name(device)} | AMP(fp16): {use_amp}", flush=True)

    # Lazy imports make troubleshooting library installs easier.
    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    records = metadata["records"]
    print(f"Fixed train subset: {len(records)} images; each uses one of its 5 captions per epoch.", flush=True)

    # Do not pass local_files_only into load_dataset: this datasets version rejects it.
    flickr = load_dataset("AnyModal/flickr30k", download_mode="reuse_dataset_if_exists")
    train_data = FlickrSubset(flickr["train"], records)
    val_data = flickr["validation"] if not args.skip_eval else None

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(
        model_id, revision="be35e1cf6ab51bb4daa2b1db53e9f543d4544b47", use_safetensors=True
    ).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,  # easy to debug, can optimize once correctness is established
        collate_fn=make_collate(processor),
        pin_memory=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["metadata"] = str(settings["metadata"])
    settings["output"] = str(settings["output"])
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Images in dataset: {len(train_data)} | batches/epoch: {len(loader)}", flush=True)

    logs = []
    global_step = 0
    best_score = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, step_count = 0.0, 0
        tick = time.time()
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if step == 1 and epoch == 1:
                print("First batch shapes:", {k: tuple(v.shape) for k, v in batch.items()}, flush=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
                # Each row j is paired with column j; exactly one image per caption.
                labels = torch.arange(batch["pixel_values"].shape[0], device=device)
                loss_i = F.cross_entropy(outputs.logits_per_image.float(), labels)
                loss_t = F.cross_entropy(outputs.logits_per_text.float(), labels)
                loss = (loss_i + loss_t) / 2
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
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
                print(f"Epoch {epoch}/{args.epochs} | step {step}/{len(loader)} | "
                      f"global {global_step} | loss={loss.item():.4f} | "
                      f"sec={time.time()-tick:.1f}", flush=True)
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
    print(f"Training log: {args.output / 'history.json'}", flush=True)
    print("Validation/test images have not contributed gradients.", flush=True)


if __name__ == "__main__":
    main()
