#!/usr/bin/env python3
"""Score semantic-overlap risk for mined Top-K image candidates.

For a query caption T and a candidate negative image J, the risk feature is:
    max_j cosine(CLIP_text(T), CLIP_text(caption_j_for_J))

HIGH RISK IS NOT PROOF OF A FALSE NEGATIVE. This script NEVER deletes samples.
Only source-image subset indices are used; no Hugging Face dataset loading or GPU.

Run from project root:
    python src/18_score_top10_caption_risk.py

Optional: copy the previously Codex-reviewed Top-1 audit CSV to
outputs/d5/hard_negative/hard_negative_audit.csv before running.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

VALID = {"good_hard_negative", "acceptable_hard_negative"}
FALSE = "false_negative"


def read_batches(path, size):
    with path.open(encoding="utf-8") as f:
        batch = []
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSONL on line {line_num}") from exc
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, names):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeds", type=Path, default=Path("outputs/d5/embeddings/text_embeds.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/d5/embeddings/metadata.json"))
    parser.add_argument("--top10", type=Path, default=Path("outputs/d5/hard_negative/hard_negative_top10.jsonl"))
    parser.add_argument("--old-audit", type=Path, default=Path("outputs/d5/hard_negative/hard_negative_audit.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/d5/threshold_audit"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be > 0")
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    records, captions = meta["records"], meta["captions"]
    owners = meta["caption_to_image"]
    num_images, num_captions = len(records), len(captions)
    if len(owners) != num_captions:
        raise ValueError("Metadata owners/captions mismatch")
    if any(len(rec["captions"]) != 5 for rec in records):
        raise ValueError("Expected exactly five captions per training image")
    if num_captions != 5 * num_images:
        raise ValueError("Text embeddings must correspond to five captions per image")
    # Verify captions and owners are aligned with the source image metadata.
    for img_idx, rec in enumerate(records):
        for j, cap in enumerate(rec["captions"]):
            text_idx = img_idx * 5 + j
            if captions[text_idx] != cap or int(owners[text_idx]) != img_idx:
                raise ValueError(f"Caption/owner order mismatch near text_idx={text_idx}")
    embeds = torch.load(args.embeds, map_location="cpu", weights_only=True)
    if embeds.ndim != 2 or embeds.shape[0] != num_captions:
        raise ValueError(f"Embedding shape {tuple(embeds.shape)} does not match metadata")
    embeds = F.normalize(embeds.float(), dim=-1)
    image_caps = embeds.reshape(num_images, 5, -1)  # [num_images, 5, dim]

    out_path = args.out / "top10_caption_risk.jsonl"
    summary_by_rank = defaultdict(list)
    top1_by_text = defaultdict(list)
    total = 0
    n_unique = set()
    with torch.no_grad(), out_path.open("w", encoding="utf-8") as output:
        for batch in read_batches(args.top10, args.batch_size):
            text_ids = torch.tensor([int(r["text_index"]) for r in batch], dtype=torch.long)
            negative_ids = torch.tensor([int(r["negative_index"]) for r in batch], dtype=torch.long)
            if ((text_ids < 0) | (text_ids >= num_captions)).any():
                raise ValueError("Bad text_index outside metadata")
            if ((negative_ids < 0) | (negative_ids >= num_images)).any():
                raise ValueError("Bad negative_index outside metadata")
            for r in batch:
                tid, neg = int(r["text_index"]), int(r["negative_index"])
                pos = int(r["positive_index"])
                rank = int(r["rank"])
                if not (1 <= rank <= 10):
                    raise ValueError(f"Unexpected rank={rank}; expected Top-10")
                if (pos != int(owners[tid]) or pos == neg or r["text"] != captions[tid]
                    or r["positive_image"] != records[pos]["filename"]
                    or r["negative_image"] != records[neg]["filename"]):
                    raise ValueError(f"Candidate is misaligned: text_index={tid}, rank={rank}")
                key = (tid, rank)
                if key in n_unique:
                    raise ValueError(f"Duplicate text_index+rank: {key}")
                n_unique.add(key)

            # [batch, 5, dim] @ [batch, dim, 1] => [batch, 5, 1]
            candidate_caps = image_caps[negative_ids]
            query_vecs = embeds[text_ids].unsqueeze(-1)
            cosines = torch.bmm(candidate_caps, query_vecs).squeeze(-1)
            max_values, max_pos = cosines.max(dim=1)

            for i, r in enumerate(batch):
                row = dict(r)
                score = float(max_values[i])
                nearest_caption_num = int(max_pos[i])
                neg = int(r["negative_index"])
                row["caption_risk"] = round(score, 6)
                row["closest_negative_caption"] = records[neg]["captions"][nearest_caption_num]
                row["closest_caption_number"] = nearest_caption_num + 1
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rank = int(r["rank"])
                summary_by_rank[rank].append(score)
                if rank == 1:
                    top1_by_text[r["text"]].append(row)
                total += 1
            if total % 50000 < len(batch) or total <= len(batch):
                print(f"Risk-scored {total:,} candidate pairs")

    expected = num_captions * 10
    if total != expected or len(n_unique) != expected:
        raise ValueError(f"Expected {expected} Top-10 pairs, got {total}")
    rank_rows = []
    for rank, scores in sorted(summary_by_rank.items()):
        s = torch.tensor(scores)
        rank_rows.append({
            "rank": rank, "num_pairs": len(scores),
            "mean_risk": round(float(s.mean()), 5),
            "median_risk": round(float(s.median()), 5),
            "p90_risk": round(float(torch.quantile(s, .9)), 5),
            "fraction_ge_080": round(float((s >= .80).float().mean()), 5),
            "fraction_ge_085": round(float((s >= .85).float().mean()), 5),
            "fraction_ge_090": round(float((s >= .90).float().mean()), 5),
        })
    write_csv(args.out / "risk_by_rank.csv", rank_rows, list(rank_rows[0]))

    if args.old_audit.exists():
        audit = load_csv(args.old_audit)
        matched, skipped = [], Counter()
        for a in audit:
            query = a.get("query_text", "")
            matches = top1_by_text.get(query, [])
            if len(matches) != 1:
                skipped["ambiguous" if matches else "not_found"] += 1
                continue
            r = matches[0]
            matched.append({
                "case_id": a.get("case_id", ""), "text_index": r["text_index"],
                "query_text": query, "positive_image": r["positive_image"],
                "negative_image": r["negative_image"], "caption_risk": r["caption_risk"],
                "closest_negative_caption": r["closest_negative_caption"],
                "quality": a.get("quality", "").strip().lower(),
                "error_type": a.get("error_type", ""),
                "reason": a.get("reason", ""),
            })
        if matched:
            write_csv(args.out / "old_top1_audit_with_risk.csv", matched, list(matched[0]))
        eligible = [r for r in matched if r["quality"] in VALID | {FALSE}]
        n_false = sum(r["quality"] == FALSE for r in eligible)
        n_valid = sum(r["quality"] in VALID for r in eligible)
        print(f"Old Top-1 audit matched {len(matched)}/{len(audit)}; skipped={dict(skipped)}")
        print(f"Comparable cases: false_negative={n_false}, valid={n_valid}")
        if n_false and n_valid:
            metrics = []
            for threshold in (.70, .75, .80, .85, .90, .95):
                tp = sum(r["quality"] == FALSE and float(r["caption_risk"]) >= threshold for r in eligible)
                fp = sum(r["quality"] in VALID and float(r["caption_risk"]) >= threshold for r in eligible)
                metrics.append({
                    "threshold": threshold, "false_detected": tp,
                    "false_total": n_false, "false_recall": round(tp/n_false, 4),
                    "valid_flagged": fp, "valid_total": n_valid,
                    "valid_false_flag_rate": round(fp/n_valid, 4),
                })
            write_csv(args.out / "pilot_top1_threshold_sweep.csv", metrics, list(metrics[0]))
            print("Pilot threshold comparison (Top-1 only; NOT a validated final threshold):")
            for r in metrics:
                print(f"  T={r['threshold']:.2f}: false recall={r['false_recall']:.1%}, "
                      f"valid false-flag={r['valid_false_flag_rate']:.1%}")
    else:
        print(f"Optional old audit not found: {args.old_audit}")
        print("To compare against the 100 existing labels, copy that CSV there and rerun.")
    print(f"Done. Full annotated Top-10: {out_path}")
    print(f"By-rank summaries: {args.out / 'risk_by_rank.csv'}")
    print("IMPORTANT: Risk score is a screening feature, not a ground-truth label.")
    print("Top-1 audit is not sufficient to validate a threshold for ranks 2-10.")


if __name__ == "__main__":
    main()
