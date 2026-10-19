"""Generated, reproducible CPU SDN flow classifier; synthetic data by default."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONFIG = {}  # GENERATED_CONFIG
CLASSES = CONFIG.get('classes', ['Benign', 'Scan', 'Flood'])


class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.nn.functional.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))


class RecurrentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(32, 32, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        y, _ = self.attention(x, x, x, need_weights=False)
        return self.norm(x + y)


class TopOneMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(32, 3)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(32, 48), nn.GELU(), nn.Linear(48, len(CLASSES))) for _ in range(3)])
        self.aux_loss = torch.tensor(0.)

    def forward(self, x):
        probs = self.router(x).softmax(-1)
        choice = probs.argmax(-1)
        output = x.new_zeros((len(x), len(CLASSES)))
        for i, expert in enumerate(self.experts):
            mask = choice == i
            if mask.any():
                output[mask] = expert(x[mask]) * probs[mask, i:i + 1]
        load = torch.nn.functional.one_hot(choice, 3).float().mean(0)
        self.aux_loss = 3 * (load * probs.mean(0)).sum()
        return output


class IDS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config['features'], 32)
        constructors = {'conv': ConvBlock, 'lstm': RecurrentBlock, 'attention': AttentionBlock}
        self.blocks = nn.ModuleList([constructors[k]() for k in config['blocks']])
        self.head = TopOneMoE() if config['head'] == 'moe' else nn.Linear(32, len(CLASSES))

    def forward(self, x):
        x = self.projection(x)
        for i, block in enumerate(self.blocks):
            try:
                x = block(x)
            except Exception as exc:
                raise RuntimeError(f'node=block_{i}: {exc}') from exc
        if self.config['pool']:
            x = x.mean(1)
        if x.ndim != 2:
            raise RuntimeError('node=head: expected [B, 32]; add mean pooling')
        return self.head(x)


def synthetic(n, rng):
    y = np.arange(n, dtype=np.int64) % 3
    rng.shuffle(y)
    x = rng.normal(0, 1, (n, CONFIG['sequence'], CONFIG['features'])).astype(np.float32)
    # Distinct but overlapping flow signatures, solely for executable smoke tests.
    x[y == 1, :, :2] += 1.25
    x[y == 2, :, 2:4] += 1.5
    return x, y


def load_data(path):
    if path:
        with np.load(path, allow_pickle=False) as data:
            parts = [(data[f'X_{s}'].astype(np.float32), data[f'y_{s}']) for s in ['train', 'val', 'test']]
    else:
        rng = np.random.default_rng(CONFIG['seed'])
        parts = [synthetic(n, rng) for n in [360, 120, 180]]
    for x, y in parts:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CONFIG['sequence'], CONFIG['features']):
            raise ValueError('node=input: X shape must be [N, sequence, features]')
        if y.shape != (len(x),) or len(x) < len(CLASSES) or y.dtype.kind not in 'iu' or not np.isin(y, range(len(CLASSES))).all() or not np.isfinite(x).all():
            raise ValueError('node=input: invalid labels, empty split or non-finite features')
        if len(np.unique(y)) != len(CLASSES):
            raise ValueError('node=input: each split must contain all configured classes')
    mean = parts[0][0].mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = parts[0][0].std(axis=(0, 1), keepdims=True, dtype=np.float64).clip(1e-6).astype(np.float32)
    np.savez('scaler.npz', mean=mean, std=std)
    return [(torch.from_numpy((x - mean) / std), torch.from_numpy(y.astype(np.int64))) for x, y in parts]


def metrics(y, pred):
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    tp = np.diag(confusion)
    recall = tp / np.maximum(confusion.sum(1), 1)
    precision = tp / np.maximum(confusion.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {'f1_macro': float(f1.mean()), 'recall': dict(zip(CLASSES, recall.tolist())), 'confusion': confusion.tolist(), 'accuracy': float(tp.sum() / len(y))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='NPZ with pre-windowed, independent train/val/test splits')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    data_path = args.data or ('flows.npz' if CONFIG.get('data_source') == 'csv' else None)
    train, val, test = load_data(data_path)
    model = IDS(CONFIG)
    optimizer_cls = torch.optim.RAdam if CONFIG['optimizer'] == 'radam' else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(*train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(CONFIG['seed']))
    history, best, best_score = [], None, -1
    for epoch in range(CONFIG['epochs']):
        model.train()
        total = 0.
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if CONFIG['head'] == 'moe':
                loss = loss + 0.01 * model.head.aux_loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
        model.eval()
        with torch.inference_mode():
            pred = predict(model, val[0])
        score = metrics(val[1].numpy(), pred)['f1_macro']
        history.append({'epoch': epoch + 1, 'loss': total / len(train[1]), 'val_f1': score})
        print(json.dumps(history[-1]), flush=True)
        if score > best_score:
            best_score, best = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        pred = predict(model, test[0])
        result = metrics(test[1].numpy(), pred)
        if 'latency' in CONFIG['metrics']:
            for i in range(30):
                model(test[0][i % len(test[0]): i % len(test[0]) + 1])
            durations = []
            for i in range(200):
                x = test[0][i % len(test[0]): i % len(test[0]) + 1]
                start = time.perf_counter_ns()
                model(x)
                durations.append((time.perf_counter_ns() - start) / 1e6)
            result['latency_ms'] = dict(zip(['p50', 'p95', 'p99'], np.percentile(durations, [50, 95, 99]).tolist()))
    # Validation F1 remains available for checkpoint selection, regardless of display selection.
    for key in ['f1_macro', 'recall']:
        if key not in CONFIG['metrics']:
            result.pop(key)
    result.update(history=history, parameters=sum(p.numel() for p in model.parameters()),
                  samples={'train': len(train[1]), 'validation': len(val[1]), 'test': len(test[1])},
                  data_source='uploaded_csv' if CONFIG.get('data_source') == 'csv' else ('real_npz' if args.data else 'synthetic_smoke_test'),
                  classes=CLASSES,
                  environment={'python': platform.python_version(), 'torch': torch.__version__,
                               'device': 'CPU', 'processor': platform.processor() or platform.machine(),
                               'threads': 2, 'batch_size': 1, 'warmup': 30, 'timed_samples': 200,
                               'latency_scope': 'model forward only; excludes preprocessing and network'},
                  config=CONFIG)
    if CONFIG.get('data_source') == 'csv':
        result['dataset'] = json.loads(Path('dataset_report.json').read_text(encoding='utf-8'))
    torch.save({'state_dict': model.state_dict(), 'config': CONFIG}, 'model.pt')
    Path('results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Completed: results.json, model.pt, scaler.npz', flush=True)


def predict(model, x):
    return np.concatenate([model(batch).argmax(-1).numpy() for batch in x.split(256)])


if __name__ == '__main__':
    main()
