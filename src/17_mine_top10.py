"""Mine top-K image negatives per caption from cached CLIP embeddings.

Example:
    python src/17_mine_top10.py

Each JSONL line stores one query / candidate-image pair. Indices are relative
 to the 5,000-image subset (metadata['records']), not the full HF train split.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Mine Top-K CLIP text-to-image candidates")
    parser.add_argument("--embed-dir", type=Path, default=Path("outputs/d5/embeddings"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/d5/hard_negative/hard_negative_top10.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    meta = json.loads((args.embed_dir / "metadata.json").read_text(encoding="utf-8"))
    records = meta["records"]
    captions = meta["captions"]
    caption_to_image = meta["caption_to_image"]

    image_embeds = torch.load(args.embed_dir / "image_embeds.pt", map_location="cpu", weights_only=True)
    text_embeds = torch.load(args.embed_dir / "text_embeds.pt", map_location="cpu", weights_only=True)

    num_images = len(records)
    num_texts = len(captions)
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("Embeddings must be 2D tensors")
    if image_embeds.shape[0] != num_images or text_embeds.shape[0] != num_texts:
        raise ValueError("Embedding rows do not match metadata records/captions")
    if image_embeds.shape[1] != text_embeds.shape[1]:
        raise ValueError("Image/text embedding dimensions differ")
    if len(caption_to_image) != num_texts:
        raise ValueError("caption_to_image length does not match captions")
    if not 1 <= args.top_k < num_images:
        raise ValueError(f"top-k must be between 1 and {num_images - 1}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not all(0 <= int(idx) < num_images for idx in caption_to_image):
        raise ValueError("Some caption_to_image indices are outside the image subset")

    # Re-normalizing protects against accidentally using non-normalized cached features.
    image_embeds = F.normalize(image_embeds.float(), dim=-1)
    text_embeds = F.normalize(text_embeds.float(), dim=-1)
    pos_indices = torch.tensor(caption_to_image, dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    first_example = None
    with torch.no_grad(), args.output.open("w", encoding="utf-8") as output:
        for start in range(0, num_texts, args.batch_size):
            stop = min(start + args.batch_size, num_texts)
            # [B, D] @ [D, N] -> [B, N]
            scores = text_embeds[start:stop] @ image_embeds.T

            # The paired original image cannot be a mined negative.
            batch_rows = torch.arange(stop - start)
            scores[batch_rows, pos_indices[start:stop]] = float("-inf")

            values, indices = torch.topk(scores, k=args.top_k, dim=1, largest=True, sorted=True)

            for local_row in range(stop - start):
                text_idx = start + local_row
                pos_idx = int(pos_indices[text_idx])
                for rank in range(args.top_k):
                    neg_idx = int(indices[local_row, rank])
                    item = {
                        "text_index": text_idx,
                        "text": captions[text_idx],
                        "positive_index": pos_idx,
                        "positive_image": records[pos_idx]["filename"],
                        "rank": rank + 1,
                        "negative_index": neg_idx,
                        "negative_image": records[neg_idx]["filename"],
                        "similarity": float(values[local_row, rank]),
                    }
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_written += 1
                    if first_example is None:
                        first_example = item
            if start == 0 or stop == num_texts or (stop // args.batch_size) % 20 == 0:
                print(f"Processed {stop:,}/{num_texts:,} captions; saved {total_written:,} pairs")

    print(f"Images: {num_images:,}; Captions: {num_texts:,}; Top-K: {args.top_k}")
    print(f"Saved: {args.output} ({total_written:,} candidate pairs)")
    print("First row:", json.dumps(first_example, ensure_ascii=False, indent=2))
    counts = Counter(caption_to_image)
    print(f"Captions per image: {dict(Counter(counts.values()))}")
    print("Important: image indices refer to metadata['records'] (subset), not full dataset.")


if __name__ == "__main__":
    main()
