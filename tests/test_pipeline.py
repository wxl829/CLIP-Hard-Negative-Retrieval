"""CPU regression checks with synthetic data; no dataset or weights downloaded."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / 'src' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    def test_mining_grouping_and_index_alignment(self):
        builder = load_script('20_build_grit_batches.py')
        trainer = load_script('21_train_clip_grouped.py')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = [{'index': i, 'filename': f'{i}.png',
                        'captions': [f'image {i} caption {j}' for j in range(5)]}
                       for i in range(12)]
            metadata = {'records': records,
                        'captions': [c for r in records for c in r['captions']],
                        'caption_to_image': [i for i in range(12) for _ in range(5)]}
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            torch.manual_seed(42)
            images = F.normalize(torch.randn(12, 512), dim=-1)
            texts = images.repeat_interleave(5, dim=0)
            torch.save(images, directory / 'image_embeds.pt')
            torch.save(texts, directory / 'text_embeds.pt')
            candidates = directory / 'top10.jsonl'
            subprocess.run([sys.executable, str(ROOT / 'src/17_mine_top10.py'),
                            '--embed-dir', str(directory), '--output', str(candidates)],
                           check=True, capture_output=True, text=True)
            rows = [json.loads(s) for s in candidates.read_text().splitlines()]
            self.assertEqual(len(rows), 600)
            self.assertTrue(all(r['negative_index'] != r['positive_index'] for r in rows))
            _, captions, adjacency, neighbors, img, txt = builder.read_inputs(directory, candidates, 10)
            random_batches = builder.make_random_batches(12, 4, 42)
            batches = builder.make_grouped_batches(adjacency, 4, 42)
            self.assertEqual(batches, builder.make_grouped_batches(adjacency, 4, 42))
            builder.check_coverage(batches, 12, 4, 'synthetic')
            choices = builder.choose_captions(captions, 42, 1)
            stats = builder.measure(batches, choices, img, txt, neighbors)
            self.assertEqual(stats['evaluated_queries'], 12)
            self.assertTrue(0 <= stats['query_with_top10_neighbor_fraction'] <= 1)
            info = {'epoch': 1, 'selected_caption_indices': choices,
                    'random_batches': random_batches, 'grit_batches': batches}
            payload = {'num_images': 12, 'batch_size': 4, 'seed': 42, 'epochs': [info]}
            plan = directory / 'groups.json'
            plan.write_text(json.dumps(payload))
            self.assertEqual(trainer.load_checked_groups(plan, metadata, 4, 1), [info])
            info['selected_caption_indices'][0] = 5  # belongs to another image
            plan.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                trainer.load_checked_groups(plan, metadata, 4, 1)

    def test_mask_orientation_and_finite_gradients(self):
        masking = load_script('23_train_grit_blip_mask.py')
        pairs = {(20, 100)}
        mask = masking.build_batch_mask([10, 20], [100, 200], pairs, 'cpu')
        self.assertEqual(mask.tolist(), [[False, False], [True, False]])
        logits = torch.tensor([[2., 1.], [1., 2.]], requires_grad=True)
        labels = torch.arange(2)
        loss = (F.cross_entropy(logits.masked_fill(mask, -torch.inf), labels)
                + F.cross_entropy(logits.T.masked_fill(mask.T, -torch.inf), labels)) / 2
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            masking.build_batch_mask([10, 20], [100, 200], {(10, 100)}, 'cpu')
        groups = [{'selected_caption_indices': [0, 5], 'grit_batches': [[0, 1]]}]
        self.assertEqual(masking.scheduled_mask_hits(groups, 'grit', {(1, 0)}, [0]*5+[1]*5), [1])

    def test_recall_protocol_on_perfect_synthetic_features(self):
        trainer = load_script('21_train_clip_grouped.py')

        class Image:
            def __init__(self, index):
                self.index = index

            def convert(self, _):
                return self

        class Validation:
            def __len__(self):
                return 12

            def __getitem__(self, key):
                ids = list(range(12))[key]
                return {'image': [Image(i) for i in ids],
                        'original_alt_text': [[str(i)]*5 for i in ids]}

        class Processor:
            def __call__(self, images=None, text=None, **kwargs):
                if images is not None:
                    return {'pixel_values': torch.eye(12)[[im.index for im in images]]}
                return {'input_ids': torch.eye(12)[[int(t) for t in text]]}

        class Model:
            def eval(self):
                return self

            def get_image_features(self, pixel_values):
                return pixel_values

            def get_text_features(self, input_ids):
                return input_ids

        metrics = trainer.evaluate(Model(), Processor(), Validation(), 'cpu', 4, 10)
        self.assertEqual(len(metrics), 6)
        self.assertTrue(all(value == 100 for value in metrics.values()))

    def test_published_results_match_histories(self):
        spec = importlib.util.spec_from_file_location('summary', ROOT / 'summarize_grit_seeds.py')
        summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summary)
        with (ROOT / 'outputs/d7/seed_summary.csv').open(newline='') as file:
            for row in csv.DictReader(file):
                metrics = summary.load(ROOT / 'outputs/d7', int(row['seed']), row['sampler'], 10)
                for metric in summary.METRICS:
                    self.assertEqual(float(row[metric]), metrics[metric])


if __name__ == '__main__':
    unittest.main()
