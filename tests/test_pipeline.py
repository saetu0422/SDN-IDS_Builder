import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class PipelineTests(unittest.TestCase):
    def test_invalid_connection_identifies_head(self):
        report = app.inspect(app.normalize({**app.DEFAULT, 'pool': False}))
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors'][0]['node'], 'head')

    def test_reject_invalid_configuration(self):
        for patch in [{'features': 0}, {'blocks': ['unknown']}, {'pool': 'false'}, {'metrics': ['fake']}, {'head': 'unknown'}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                app.normalize({**app.DEFAULT, **patch})

    def test_generated_code_and_real_training_variants(self):
        for head, blocks in [('dense', ['conv', 'lstm', 'attention']), ('moe', ['attention', 'conv']), ('dense', [])]:
            with self.subTest(head=head, blocks=blocks), tempfile.TemporaryDirectory() as temp:
                c = app.normalize({**app.DEFAULT, 'head': head, 'blocks': blocks, 'epochs': 1})
                folder = Path(temp)
                app.write_project(c, folder)
                ast.parse((folder / 'train.py').read_text(encoding='utf-8'))
                p = subprocess.run([sys.executable, str(folder / 'train.py')], cwd=folder, capture_output=True, text=True, timeout=90)
                self.assertEqual(p.returncode, 0, p.stderr)
                result = json.loads((folder / 'results.json').read_text())
                self.assertEqual(sum(sum(row) for row in result['confusion']), 180)
                self.assertTrue(0 <= result['f1_macro'] <= 1)
                self.assertLessEqual(result['latency_ms']['p50'], result['latency_ms']['p95'])
                self.assertLessEqual(result['latency_ms']['p95'], result['latency_ms']['p99'])
                self.assertTrue((folder / 'model.pt').exists())
                self.assertTrue((folder / 'scaler.npz').exists())

    def test_metrics_and_npz_contract(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            app.write_project(app.normalize({**app.DEFAULT, 'epochs': 1, 'blocks': [], 'metrics': []}), folder)
            spec = importlib.util.spec_from_file_location('generated_test', folder / 'train.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
            self.assertEqual(result['f1_macro'], 1.)
            rng = np.random.default_rng(123)
            parts = {f'{kind}_{split}': value for split in ['train', 'val', 'test']
                     for kind, value in zip(['X', 'y'], module.synthetic(12, rng))}
            data = folder / 'flows.npz'
            np.savez(data, **parts)
            p = subprocess.run([sys.executable, str(folder / 'train.py'), '--data', str(data)], cwd=folder, capture_output=True, text=True, timeout=90)
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads((folder / 'results.json').read_text())
            self.assertEqual(r['data_source'], 'real_npz')
            self.assertNotIn('latency_ms', r)
            self.assertNotIn('recall', r)
            self.assertNotIn('f1_macro', r)
            parts['X_test'] = np.zeros((12, 2, 4), dtype=np.float32)
            np.savez(data, **parts)
            p = subprocess.run([sys.executable, str(folder / 'train.py'), '--data', str(data)], cwd=folder, capture_output=True, text=True, timeout=90)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('node=input', p.stderr)


if __name__ == '__main__':
    unittest.main()
