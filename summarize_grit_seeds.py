#!/usr/bin/env python3
"""Summarize matched Random vs GRIT best-checkpoint validation results, seeds 42/123/2026.
Run from the project root. Writes outputs/d7/seed_summary.csv.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = ('T2I_R@1', 'I2T_R@1', 'T2I_R@5', 'I2T_R@5', 'T2I_R@10', 'I2T_R@10')


def path_for(root, seed, sampler):
    if seed == 42:
        return root / ('ft_random_matched' if sampler == 'random' else 'ft_grit_style')
    return root / f'ft_{sampler}_seed{seed}'


def load(root, seed, sampler, expected_epochs):
    directory = path_for(root, seed, sampler)
    mfile = directory / 'best' / 'metrics.json'
    hfile = directory / 'history.json'
    if not mfile.exists() or not hfile.exists():
        raise FileNotFoundError(f'Incomplete/missing run at {directory}: need best/metrics.json and history.json')
    h = json.loads(hfile.read_text(encoding='utf-8'))
    if len(h) != expected_epochs:
        raise ValueError(f'{directory}: expected {expected_epochs} epochs, got {len(h)}')
    m = json.loads(mfile.read_text(encoding='utf-8'))
    best = max(h, key=lambda row: row['T2I_R@1'] + row['I2T_R@1'])
    if m != best:
        raise ValueError(f'{directory}: best metrics do not match the best epoch in history')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config['seed'] != seed or config['sampler'] != sampler or config['skip_eval']:
        raise ValueError(f'{directory}: config does not match the requested run')
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('outputs/d7'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2026])
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate seeds are not allowed')

    data, rows = {}, []
    for seed in args.seeds:
        data[seed] = {}
        for sampler in ('random', 'grit'):
            m = load(args.root, seed, sampler, args.epochs)
            data[seed][sampler] = m
            row = {'seed': seed, 'sampler': sampler, 'best_epoch': m['epoch']}
            row.update({metric: m[metric] for metric in METRICS})
            row['mean_R@1'] = round((m['T2I_R@1'] + m['I2T_R@1']) / 2, 4)
            rows.append(row)

    output = args.root / 'seed_summary.csv'
    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Per-run summary saved: {output}\n')

    all_metrics = METRICS + ('mean_R@1',)
    for seed in args.seeds:
        r, g = data[seed]['random'], data[seed]['grit']
        print(f'Seed {seed} | best epoch: Random={r["epoch"]}, GRIT={g["epoch"]}')
        for metric in all_metrics:
            rv = (r['T2I_R@1'] + r['I2T_R@1']) / 2 if metric == 'mean_R@1' else r[metric]
            gv = (g['T2I_R@1'] + g['I2T_R@1']) / 2 if metric == 'mean_R@1' else g[metric]
            print(f'  {metric:<11} Random={rv:8.4f}%  GRIT={gv:8.4f}%  Δ={gv-rv:+.4f} pp')
        print()

    print('Across-seed mean ± sample SD (percentage points):')
    for metric in all_metrics:
        rv = [((data[s]['random']['T2I_R@1'] + data[s]['random']['I2T_R@1']) / 2
              if metric == 'mean_R@1' else data[s]['random'][metric]) for s in args.seeds]
        gv = [((data[s]['grit']['T2I_R@1'] + data[s]['grit']['I2T_R@1']) / 2
              if metric == 'mean_R@1' else data[s]['grit'][metric]) for s in args.seeds]
        differences = [g-r for g, r in zip(gv, rv)]
        sd = lambda seq: statistics.stdev(seq) if len(seq) > 1 else float('nan')
        print(f'{metric:<11} Random {statistics.mean(rv):.4f}±{sd(rv):.4f}; '
              f'GRIT {statistics.mean(gv):.4f}±{sd(gv):.4f}; '
              f'paired Δ {statistics.mean(differences):+.4f}±{sd(differences):.4f} pp')
    print('\nPaired seeds reduce noise, but three seeds alone do not establish statistical significance.')


if __name__ == '__main__':
    main()
