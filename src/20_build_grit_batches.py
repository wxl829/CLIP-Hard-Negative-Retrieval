#!/usr/bin/env python3
"""GRIT-inspired STATIC grouping using the existing Flickr30k CLIP Top-10 graph.

This is a data preparation / diagnostics script: it NEVER trains CLIP.
Requires: outputs/d5/embeddings/{metadata.json,image_embeds.pt,text_embeds.pt}
          outputs/d5/hard_negative/hard_negative_top10.jsonl

Example:
  python src/20_build_grit_batches.py --batch-size 16 --epochs 1 --seed 42
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def read_inputs(embed_dir: Path, top10_file: Path, k: int):
    metadata = json.loads((embed_dir / "metadata.json").read_text(encoding="utf-8"))
    records = metadata["records"]
    captions = metadata["captions"]
    owner = metadata["caption_to_image"]
    n_images, n_texts = len(records), len(captions)
    assert len(owner) == n_texts, "caption_to_image 长度与 captions 不一致"
    caption_ids_by_image = [[] for _ in range(n_images)]
    for tid, iid in enumerate(owner):
        assert 0 <= iid < n_images, f"Invalid image index for text {tid}: {iid}"
        caption_ids_by_image[iid].append(tid)
    assert all(len(x) == 5 for x in caption_ids_by_image), "当前方案要求每张图片恰好五条 Caption"

    # Graph vertices are SUBSET image indices, not original Flickr30k dataset indices.
    adjacency = [defaultdict(float) for _ in range(n_images)]
    topk_by_text = [set() for _ in range(n_texts)]
    seen_rank_masks = [0] * n_texts
    count = 0
    with top10_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            tid = int(item["text_index"])
            pos = int(item["positive_index"])
            neg = int(item["negative_index"])
            rank = int(item["rank"])
            assert 0 <= tid < n_texts and 0 <= neg < n_images, f"Bad index at line {line_no}"
            assert pos == owner[tid] and neg != pos, f"Wrong positive/negative mapping at line {line_no}"
            assert item["positive_image"] == records[pos]["filename"], f"Wrong positive filename at line {line_no}"
            assert item["negative_image"] == records[neg]["filename"], f"Wrong negative filename at line {line_no}"
            assert 1 <= rank <= k, f"Unexpected rank at line {line_no}"
            bit = 1 << (rank - 1)
            assert (seen_rank_masks[tid] & bit) == 0, f"Duplicate rank for text {tid}"
            seen_rank_masks[tid] |= bit
            assert neg not in topk_by_text[tid], f"Duplicate negative for text {tid}"
            topk_by_text[tid].add(neg)
            # Stronger / repeated Top-K relations receive more weight.
            # Symmetrize for image-level batch construction. NOT the original GRIT formula.
            w = (k + 1 - rank) / k
            adjacency[pos][neg] += w
            adjacency[neg][pos] += w
            count += 1
    assert count == n_texts * k, f"Expected {n_texts*k} candidates, got {count}"
    assert all(mask == (1 << k) - 1 for mask in seen_rank_masks), "Some captions lack a complete Top-K"

    img = torch.load(embed_dir / "image_embeds.pt", map_location="cpu", weights_only=True).float()
    txt = torch.load(embed_dir / "text_embeds.pt", map_location="cpu", weights_only=True).float()
    assert tuple(img.shape) == (n_images, 512), f"Unexpected image shape: {img.shape}"
    assert tuple(txt.shape) == (n_texts, 512), f"Unexpected text shape: {txt.shape}"
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    return records, caption_ids_by_image, adjacency, topk_by_text, img, txt


def make_random_batches(n: int, batch_size: int, seed: int):
    items = list(range(n))
    random.Random(seed).shuffle(items)
    return [items[i:i + batch_size] for i in range(0, n, batch_size)]


def make_grouped_batches(adjacency, batch_size: int, seed: int):
    """Greedy graph grouping, without replacement, one appearance per image/epoch."""
    rng = random.Random(seed)
    unused = set(range(len(adjacency)))
    batches = []
    while unused:
        start = rng.choice(sorted(unused))
        unused.remove(start)
        batch = [start]
        while len(batch) < batch_size and unused:
            # Score all remaining neighbors by affinity to CURRENT batch members.
            scores = defaultdict(float)
            for chosen in batch:
                for candidate, weight in adjacency[chosen].items():
                    if candidate in unused:
                        scores[candidate] += weight
            if scores:
                highest = max(scores.values())
                tied = [idx for idx, score in scores.items() if abs(score - highest) < 1e-10]
                next_image = rng.choice(sorted(tied))
            else:
                # Graph disconnected / candidate neighborhood exhausted.
                next_image = rng.choice(sorted(unused))
            unused.remove(next_image)
            batch.append(next_image)
        batches.append(batch)
    return batches


def check_coverage(batches, n_images, batch_size, name):
    flat = [idx for batch in batches for idx in batch]
    assert len(flat) == n_images and set(flat) == set(range(n_images)), (
        f"{name}: missing/repeated image indexes"
    )
    assert all(len(batch) == len(set(batch)) and 1 <= len(batch) <= batch_size for batch in batches)
    assert all(len(batch) == batch_size for batch in batches[:-1]), f"{name}: short batch before final"


def choose_captions(caption_ids_by_image, seed: int, epoch: int):
    """Same per-image caption choice for random and grouped evaluation."""
    return [
        random.Random(seed + epoch * 1_000_003 + iid).choice(cap_ids)
        for iid, cap_ids in enumerate(caption_ids_by_image)
    ]


@torch.no_grad()
def measure(batches, caption_choices, img_embeds, txt_embeds, topk_by_text):
    sim_sum, sim_count = 0.0, 0
    hardest_sum, query_count = 0.0, 0
    topk_hits, with_any, available_pairs = 0, 0, 0
    for batch in batches:
        if len(batch) < 2:
            continue
        tids = [caption_choices[iid] for iid in batch]
        # rows=query texts, columns=images; [B,512] @ [512,B] -> [B,B]
        sims = txt_embeds[tids] @ img_embeds[batch].T
        b = len(batch)
        diagonal = torch.eye(b, dtype=torch.bool)
        negatives = sims[~diagonal]
        sim_sum += negatives.sum().item()
        sim_count += negatives.numel()
        hardest_sum += sims.masked_fill(diagonal, -1e9).max(dim=1).values.sum().item()
        query_count += b
        for row, tid in enumerate(tids):
            hits = sum(
                int(other in topk_by_text[tid])
                for col, other in enumerate(batch) if col != row
            )
            topk_hits += hits
            with_any += int(hits > 0)
            available_pairs += b - 1
    return {
        "offdiag_mean_cosine": round(sim_sum / sim_count, 6),
        "per_query_max_negative_cosine": round(hardest_sum / query_count, 6),
        "top10_inbatch_pair_count": topk_hits,
        "top10_inbatch_pair_fraction": round(topk_hits / available_pairs, 6),
        "query_with_top10_neighbor_fraction": round(with_any / query_count, 6),
        "evaluated_queries": query_count,
    }


def main():
    p = argparse.ArgumentParser(description="Build and inspect GRIT-inspired Top-10 grouped batches, NO training")
    p.add_argument("--embed-dir", type=Path, default=Path("outputs/d5/embeddings"))
    p.add_argument("--top10", type=Path, default=Path("outputs/d5/hard_negative/hard_negative_top10.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/d7/grit_batches"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    if args.batch_size < 2 or args.epochs < 1 or args.k < 1:
        p.error("Require batch-size >=2, epochs >=1 and k >=1")
    records, captions_by_img, adjacency, topk_by_text, img, txt = read_inputs(args.embed_dir, args.top10, args.k)
    n = len(records)
    n_batches = math.ceil(n / args.batch_size)
    print(f"Loaded {n:,} images, {len(topk_by_text):,} captions; {args.k} Top-K per caption")
    print(f"Embedding shapes: image={tuple(img.shape)} text={tuple(txt.shape)}")
    print(f"Batch size={args.batch_size}, batches/epoch={n_batches}; no model training")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_epochs = []
    rows = []
    for epoch in range(1, args.epochs + 1):
        # Same seed across samplers for reproducibility, without claiming identical ordering.
        epoch_seed = args.seed + epoch * 1009
        rand_batches = make_random_batches(n, args.batch_size, epoch_seed)
        grit_batches = make_grouped_batches(adjacency, args.batch_size, epoch_seed)
        check_coverage(rand_batches, n, args.batch_size, "random")
        check_coverage(grit_batches, n, args.batch_size, "grouped")
        captions = choose_captions(captions_by_img, args.seed, epoch)
        random_stats = measure(rand_batches, captions, img, txt, topk_by_text)
        grit_stats = measure(grit_batches, captions, img, txt, topk_by_text)
        all_epochs.append({
            "epoch": epoch,
            "selected_caption_indices": captions,
            "random_batches": rand_batches,
            "grit_batches": grit_batches,
        })
        for method, metrics in (("random", random_stats), ("grit_style", grit_stats)):
            rows.append({"epoch": epoch, "sampler": method, **metrics})
        print(f"\nEpoch {epoch}: images exactly once in each sampler; {len(grit_batches)} batches")
        for name, m in (("Random", random_stats), ("GRIT-style", grit_stats)):
            print(f"  {name:10s} mean_other={m['offdiag_mean_cosine']:.4f}, "
                  f"hardest={m['per_query_max_negative_cosine']:.4f}, "
                  f"Top10_pairs={m['top10_inbatch_pair_count']}, "
                  f"queries_with_Top10={m['query_with_top10_neighbor_fraction']:.2%}")
    payload = {
        "note": "Static GRIT-inspired grouping, NOT original GRIT-VLP; no CLIP training, no false-negative filtering.",
        "num_images": n,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "top_k": args.k,
        "epochs": all_epochs,
    }
    group_file = args.output_dir / "grouped_batches.json"
    group_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_file = args.output_dir / "batch_quality.csv"
    with summary_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {group_file}")
    print(f"Saved: {summary_file}")
    print("First GRIT-style batch image indexes:", all_epochs[0]["grit_batches"][0])
    print("First GRIT-style batch filenames:", [records[i]["filename"] for i in all_epochs[0]["grit_batches"][0]])
    print("IMPORTANT: indexes refer to metadata['records'] (5,000-image SUBSET), not full dataset.")


if __name__ == "__main__":
    main()
