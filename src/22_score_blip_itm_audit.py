#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 D8 固定抽样的候选图文对执行 BLIP-ITM 推理，并可与 Codex 初筛标签比较。

首次小规模验证：
  CUDA_VISIBLE_DEVICES=2 python src/22_score_blip_itm_audit.py --max-cases 8 \
      --output outputs/d8/blip_itm_smoke.csv

完整评分：
  CUDA_VISIBLE_DEVICES=2 python src/22_score_blip_itm_audit.py

将 codex_labels.csv 传回服务器后，独立分析（不加载 GPU 模型）：
  python src/22_score_blip_itm_audit.py --analyze-only \
      --labels outputs/d8/codex_audit_384/codex_labels.csv

依赖：torch, transformers, datasets, Pillow；模型权重需事先可通过 Hugging Face
获取，或通过 --model-path 指向已下载的本地完整 BLIP 模型目录。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

LABELS = {'valid_negative', 'possible_false_negative', 'uncertain'}
SCORE_FIELDS = [
    'case_id', 'candidate_index', 'text_index', 'query_text', 'candidate_image',
    'blip_match_score', 'blip_not_match_score', 'blip_margin',
]


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmpname, path)
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)


def validate_cases(cases, metadata):
    records = metadata['records']
    captions = metadata['captions']
    owners = metadata['caption_to_image']
    if not isinstance(cases, list) or not cases:
        raise ValueError('sample_cases.json 必须是非空案例数组')
    seen = set()
    for case in cases:
        cid = str(case['case_id'])
        if cid in seen:
            raise ValueError(f'重复 case_id: {cid}')
        seen.add(cid)
        ci = int(case['candidate_index'])
        pi = int(case['positive_index'])
        ti = int(case['text_index'])
        if not (0 <= ci < len(records) and 0 <= pi < len(records) and 0 <= ti < len(captions)):
            raise ValueError(f'索引越界: {cid}')
        if pi == ci or int(owners[ti]) != pi:
            raise ValueError(f'正/负样本关系不合法: {cid}')
        if case['query_text'] != captions[ti]:
            raise ValueError(f'查询文本与 metadata 不一致: {cid}')
        if case['candidate_image'] != records[ci]['filename']:
            raise ValueError(f'候选图片文件名不一致: {cid}')
        if case['positive_image'] != records[pi]['filename']:
            raise ValueError(f'正样本图片文件名不一致: {cid}')
    return records


def score_rows(args, cases, records):
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForImageTextRetrieval

    if args.image_dir:
        image_dir = args.image_dir
        if not image_dir.is_dir():
            raise FileNotFoundError(f'找不到图片文件夹: {image_dir}')
        hf_train = None
        print(f'图片来源：导出的 JPEG 文件 {image_dir}（注意可能是缩略图）', flush=True)
    else:
        from datasets import load_dataset
        hf_train = load_dataset('AnyModal/flickr30k', download_mode='reuse_dataset_if_exists')['train']
        image_dir = None
        print(f'图片来源：Flickr30k 原始 train 数据集，共 {len(hf_train)} 张图片', flush=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'设备={device}，BLIP模型={args.model_path}，批大小={args.batch_size}', flush=True)
    processor = BlipProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model = BlipForImageTextRetrieval.from_pretrained(
        args.model_path, local_files_only=args.local_files_only,
        use_safetensors=True,
    ).to(device).eval()

    def get_image(case):
        if image_dir is not None:
            fp = image_dir / f"{case['case_id']}.jpg"
            if not fp.is_file():
                raise FileNotFoundError(f'缺少候选图片 {fp}')
            with Image.open(fp) as im:
                return im.convert('RGB')
        ci = int(case['candidate_index'])
        origin_idx = int(records[ci]['index'])
        if not 0 <= origin_idx < len(hf_train):
            raise ValueError(f'原始 train 索引越界：{case["case_id"]}={origin_idx}')
        row = hf_train[origin_idx]
        if row['filename'] != records[ci]['filename']:
            raise ValueError(f'图片索引映射错误：{case["case_id"]}, '
                             f'{row["filename"]} != {records[ci]["filename"]}')
        return row['image'].convert('RGB')

    existing = {}
    if args.output.is_file():
        for row in read_csv(args.output):
            cid = row.get('case_id')
            if cid in existing:
                raise ValueError(f'已有评分文件重复 case_id: {cid}')
            existing[cid] = row
        # Skip only if previously saved case metadata agrees, avoiding accidental mixing.
        by_id = {c['case_id']: c for c in cases}
        for cid, row in existing.items():
            if cid not in by_id:
                raise ValueError(f'评分文件存在不属于本次样本的案例: {cid}')
            c = by_id[cid]
            if (str(row['candidate_index']) != str(c['candidate_index']) or
                str(row['text_index']) != str(c['text_index']) or
                row['query_text'] != c['query_text'] or
                row['candidate_image'] != c['candidate_image']):
                raise ValueError(f'历史评分与当前样本不一致: {cid}')
        print(f'已有评分 {len(existing)} 条，将跳过并继续未完成案例', flush=True)

    todo = [c for c in cases if c['case_id'] not in existing]
    if args.max_cases:
        todo = todo[:args.max_cases]
    if not todo:
        print('没有待评分案例', flush=True)
        return

    @torch.inference_mode()
    def infer(batch_cases):
        imgs = [get_image(c) for c in batch_cases]
        texts = [c['query_text'] for c in batch_cases]
        inputs = processor(
            images=imgs, text=texts, return_tensors='pt', padding=True,
            truncation=True, max_length=77,
        )
        inputs = inputs.to(device)
        if device.type == 'cuda' and args.amp:
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(**inputs, use_itm_head=True)
        else:
            output = model(**inputs, use_itm_head=True)
        logits = output.itm_score
        if logits is None or logits.ndim != 2 or logits.shape != (len(batch_cases), 2):
            raise RuntimeError(f'预期 ITM logits 为 [B,2]，实际 {getattr(logits, "shape", None)}')
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        log = logits.float().cpu().tolist()
        return probs, log

    processed = 0
    for i in range(0, len(todo), args.batch_size):
        subset = todo[i:i + args.batch_size]
        probs, log = infer(subset)
        for case, pp, zz in zip(subset, probs, log):
            if not all(math.isfinite(v) for v in (*pp, *zz)):
                raise RuntimeError(f'非有限 BLIP 输出: {case["case_id"]}')
            existing[case['case_id']] = {
                'case_id': case['case_id'],
                'candidate_index': case['candidate_index'],
                'text_index': case['text_index'],
                'query_text': case['query_text'],
                'candidate_image': case['candidate_image'],
                'blip_match_score': f'{pp[1]:.8f}',
                'blip_not_match_score': f'{pp[0]:.8f}',
                'blip_margin': f'{zz[1] - zz[0]:.8f}',
            }
        ordered = [existing[c['case_id']] for c in cases if c['case_id'] in existing]
        write_csv_atomic(args.output, SCORE_FIELDS, ordered)
        processed += len(subset)
        if processed % (args.batch_size * 5) == 0 or processed == len(todo):
            print(f'BLIP已评分：本次{processed}/{len(todo)}，累计{len(ordered)}/{len(cases)}；已保存 {args.output}', flush=True)
    print('完成。匹配分数越高，仅代表 BLIP 越倾向于图文匹配，不是人工真值。', flush=True)


def analyze(score_file: Path, labels_file: Path, out_file: Path, threshold: float):
    if not score_file.is_file():
        raise FileNotFoundError(f'先完成 BLIP 评分: {score_file}')
    if not labels_file.is_file():
        raise FileNotFoundError(f'找不到 Codex 审核结果: {labels_file}')
    scores = read_csv(score_file)
    labels = read_csv(labels_file)
    if not scores or not labels:
        raise ValueError('评分或标签 CSV 为空')
    if not {'case_id', 'label'}.issubset(labels[0]):
        raise ValueError('标签 CSV 至少需要 case_id,label 两列')
    if not {'case_id', 'blip_match_score'}.issubset(scores[0]):
        raise ValueError('BLIP 评分 CSV 字段不完整')
    by_label = {}
    for row in labels:
        cid = row['case_id'].strip()
        lab = row['label'].strip().lower()
        if cid in by_label:
            raise ValueError(f'Codex 标签重复: {cid}')
        if lab not in LABELS:
            raise ValueError(f'未知标签 {cid}: {lab}')
        by_label[cid] = row

    merged = []
    tp = fp = tn = fn = 0
    missing = 0
    dist = {s: [] for s in LABELS}
    for row in scores:
        cid = row['case_id']
        labrow = by_label.get(cid)
        if labrow is None:
            missing += 1
            continue
        label = labrow['label'].strip().lower()
        s = float(row['blip_match_score'])
        predicted = 'possible_false_negative' if s >= threshold else 'valid_negative'
        is_agree = predicted == label if label != 'uncertain' else ''
        merged.append({**row, 'codex_label': label,
                       'codex_confidence': labrow.get('confidence', ''),
                       'codex_reason': labrow.get('reason', ''),
                       'predicted_at_threshold': predicted,
                       'agree_with_codex': is_agree})
        dist[label].append(s)
        if label == 'uncertain':
            continue
        if label == 'possible_false_negative':
            tp += int(s >= threshold)
            fn += int(s < threshold)
        elif label == 'valid_negative':
            fp += int(s >= threshold)
            tn += int(s < threshold)

    if not merged:
        raise RuntimeError('没有可匹配的 case_id，检查 Codex CSV 与 sample_cases.json 是否来自同批案例')
    fields = SCORE_FIELDS + ['codex_label','codex_confidence','codex_reason',
                             'predicted_at_threshold','agree_with_codex']
    write_csv_atomic(out_file, fields, merged)
    print(f'BLIP与Codex标签重合 {len(merged)} 条，缺少Codex标签 {missing} 条；输出: {out_file}')
    for lab, values in dist.items():
        if values:
            avg = sum(values)/len(values)
            sortedv = sorted(values)
            print(f'  {lab}: n={len(values)}, match_score均值={avg:.4f}, 中位数={sortedv[len(values)//2]:.4f}')
    print(f'以固定演示阈值 {threshold:g} 与 Codex 标签对照（不含 uncertain）：TP={tp}, FP={fp}, TN={tn}, FN={fn}')
    precision = tp/(tp+fp) if tp+fp else float('nan')
    recall = tp/(tp+fn) if tp+fn else float('nan')
    f1 = 2*precision*recall/(precision+recall) if (math.isfinite(precision) and math.isfinite(recall) and precision+recall) else float('nan')
    print(f'  与Codex一致性指标（不是人工真值准确率）：precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}')
    print('注意：0.5是演示阈值，不是经验证的最佳阈值；不要直接用于训练Mask。')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', type=Path, default=Path('outputs/d8/codex_audit_384/sample_cases.json'))
    ap.add_argument('--metadata', type=Path, default=Path('outputs/d5/embeddings/metadata.json'))
    ap.add_argument('--model-path', default='Salesforce/blip-itm-base-coco', help='Hugging Face模型名，或本地模型目录')
    ap.add_argument('--local-files-only', action='store_true', help='BLIP权重已缓存时禁止联网')
    ap.add_argument('--image-dir', type=Path, default=None, help='可选：candidate_images目录；默认直接加载原始Flickr30k图片')
    ap.add_argument('--output', type=Path, default=Path('outputs/d8/codex_audit_384/blip_itm_scores.csv'))
    ap.add_argument('--labels', type=Path, default=None, help='可选：服务器上的Codex标签 CSV')
    ap.add_argument('--analyze-only', action='store_true', help='仅将已有BLIP分数与Codex标签比较，不加载模型')
    ap.add_argument('--comparison-output', type=Path, default=Path('outputs/d8/codex_audit_384/blip_vs_codex.csv'))
    ap.add_argument('--threshold', type=float, default=0.5, help='仅用于演示的匹配分数阈值，非校准阈值')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-cases', type=int, default=0, help='冒烟测试只计算前N条，0=全部')
    ap.add_argument('--amp', action='store_true', help='启用fp16自动混合精度；不默认启用以简化数值检查')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    if args.batch_size < 1 or args.max_cases < 0 or not 0 <= args.threshold <= 1:
        ap.error('batch-size需>=1，max-cases需>=0，threshold需在[0,1]')

    if not args.analyze_only:
        if not args.cases.is_file():
            raise FileNotFoundError(f'未找到案例文件 {args.cases}；请先在服务器准备384条案例')
        if not args.metadata.is_file():
            raise FileNotFoundError(f'未找到 metadata {args.metadata}')
        cases = read_json(args.cases)
        records = validate_cases(cases, read_json(args.metadata))
        print(f'案例数量 {len(cases)}；meta映射检查通过', flush=True)
        score_rows(args, cases, records)

    if args.labels is not None:
        analyze(args.output, args.labels, args.comparison_output, args.threshold)
    elif args.analyze_only:
        ap.error('--analyze-only 时必须提供 --labels')
    else:
        print('Codex审核结果尚未在服务器上也没关系。之后用 --analyze-only --labels <路径> 进行对照即可。')


if __name__ == '__main__':
    main()
