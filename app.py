"""Local SDN IDS assembly studio. Python 3.11+, no web dependencies."""
import ast
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import copy
import hashlib
import shutil
import dataset
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
UPLOADS = ROOT / 'uploads'
JOBS = {}
BATCHES = {}
COMPARISONS = RUNS / 'comparisons'
LOCK = threading.Lock()
DEFAULT = {
    'name': 'SDN Flow Sentinel', 'features': 12, 'sequence': 8,
    'blocks': ['conv', 'lstm', 'attention'], 'pool': True,
    'head': 'dense', 'optimizer': 'radam', 'epochs': 3,
    'metrics': ['f1_macro', 'recall', 'latency'], 'seed': 42,
    'data_source': 'synthetic', 'dataset_id': '', 'feature_columns': [],
    'label_column': '', 'split_mode': 'stratified', 'group_column': '',
}


def get_dataset(dataset_id):
    if not isinstance(dataset_id, str) or not re.fullmatch(r'[a-f0-9]{32}', dataset_id):
        raise ValueError('CSV 데이터셋을 먼저 업로드하세요.')
    try:
        return json.loads((UPLOADS / f'{dataset_id}.json').read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ValueError('업로드한 데이터셋이 없습니다. 다시 업로드하세요.') from exc


def normalize(raw):
    if not isinstance(raw, dict):
        raise ValueError('설정은 JSON 객체여야 합니다.')
    c = {k: raw.get(k, v) for k, v in DEFAULT.items()}
    if c['data_source'] not in ['synthetic', 'csv']:
        raise ValueError('지원하지 않는 데이터 소스입니다.')
    if c['data_source'] == 'csv':
        info = get_dataset(c['dataset_id'])
        c['classes'] = dataset.validate_selection(info, c)
        c['features'], c['sequence'] = len(c['feature_columns']), 1
    for key, low, high in [('features', 1 if c['data_source'] == 'csv' else 4, 64), ('sequence', 1, 32), ('epochs', 1, 10), ('seed', 0, 99999)]:
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError(f'{key}: {low}~{high} 정수가 필요합니다.')
    if not isinstance(c['name'], str) or not 1 <= len(c['name']) <= 80:
        raise ValueError('프로젝트 이름은 1~80자여야 합니다.')
    if not isinstance(c['blocks'], list) or len(c['blocks']) > 6 or any(x not in ['conv', 'lstm', 'attention'] for x in c['blocks']):
        raise ValueError('지원하지 않는 블록 또는 블록 수 초과입니다.')
    if type(c['pool']) is not bool or c['head'] not in ['dense', 'moe'] or c['optimizer'] not in ['adam', 'radam']:
        raise ValueError('지원하지 않는 모델/학습 설정입니다.')
    if not isinstance(c['metrics'], list) or any(x not in ['f1_macro', 'recall', 'latency'] for x in c['metrics']):
        raise ValueError('지원하지 않는 평가 지표입니다.')
    return c


def inspect(c):
    classes = len(c.get('classes', ['Benign', 'Scan', 'Flood']))
    nodes = [{'id': 'input', 'label': 'Flow input', 'shape': f"[B, {c['sequence']}, {c['features']}]"},
             {'id': 'preprocess', 'label': 'Standardize / project', 'shape': f"[B, {c['sequence']}, 32]"}]
    names = {'conv': 'Conv1D + GELU', 'lstm': 'LSTM', 'attention': 'Self attention · 4 heads'}
    for i, kind in enumerate(c['blocks']):
        nodes.append({'id': f'block_{i}', 'label': names[kind], 'shape': f"[B, {c['sequence']}, 32]"})
    if c['pool']:
        nodes.append({'id': 'pool', 'label': 'Mean pooling', 'shape': '[B, 32]'})
    nodes.append({'id': 'head', 'label': 'Top-1 MoE · 3 experts' if c['head'] == 'moe' else 'Linear classifier', 'shape': f'[B, {classes}]'})
    errors = []
    if not c['pool']:
        errors.append({'node': 'head', 'code': 'SHAPE_MISMATCH', 'message': '분류기는 [B, 32] 입력이 필요하지만 이전 노드가 [B, T, 32]를 출력합니다.', 'fix': 'Mean pooling을 연결하세요.'})
    return {'nodes': nodes, 'errors': errors, 'valid': not errors, 'classes': classes,
            'notes': ['입력은 시퀀스로 묶인 flow 특성입니다. 패킷 수집·SDN 컨트롤러 연결은 포함하지 않습니다.',
                      'MoE는 top-1 라우팅으로 선택된 expert만 실행합니다. 속도 향상은 측정으로 확인해야 합니다.',
                      '합성 데이터의 성능은 실행 점검용이며 실제 침입 탐지 성능을 입증하지 않습니다.']}


def generate(c):
    template = (ROOT / 'pipeline_template.py').read_text(encoding='utf-8')
    source = template.replace('CONFIG = {}  # GENERATED_CONFIG', 'CONFIG = ' + repr(c))
    ast.parse(source)
    return source


def write_project(c, folder, shared=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'train.py').write_text(generate(c), encoding='utf-8')
    (folder / 'config.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
    if c['data_source'] == 'csv':
        if shared:
            for name in ['flows.npz', 'dataset_report.json', 'split_rows.json']:
                shutil.copyfile(shared / name, folder / name)
        else:
            import numpy as np
            info = get_dataset(c['dataset_id'])
            content = (UPLOADS / f"{c['dataset_id']}.csv").read_bytes()
            arrays, report, rows = dataset.prepare(content, info, c, c['seed'])
            np.savez_compressed(folder / 'flows.npz', **arrays)
            (folder / 'dataset_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            (folder / 'split_rows.json').write_text(json.dumps(rows), encoding='utf-8')
    (folder / 'requirements.txt').write_text('torch>=2.2\nnumpy>=1.24\n', encoding='utf-8')
    (folder / 'README.md').write_text('''# Generated SDN IDS experiment

Run `python -m pip install -r requirements.txt`, then `python train.py`.
Results are written to `results.json`. Synthetic data is a smoke test only.
For uploaded CSV projects, `python train.py` automatically loads included flows.npz
and dataset_report.json. The ZIP contains selected real data: keep it private.
CSV rows are independent flows with sequence length 1; class names come from config.
Use real pre-windowed data: `python train.py --data flows.npz`.
NPZ must contain X_train/X_val/X_test float arrays [N, sequence, features]
and y_train/y_val/y_test integer arrays [N]. Labels are zero-based class indices;
uploaded CSV class names/mapping are recorded in config.json and dataset_report.json.
The synthetic default and legacy NPZ experiments use labels 0, 1, 2.
Split flows by time/session before making windows to avoid leakage.
The scaler is fitted on training data only. Validation selects the best epoch;
test data is evaluated once afterward. Latency uses CPU, batch size 1, warmup
and repeated samples; it excludes preprocessing, transport and controller overhead.
Synthetic classes: 0 benign, 1 scan, 2 flood. No SDN controller integration is included.
Top-1 MoE executes selected experts and uses an auxiliary load balancing loss.
This project does not implement SparseGPT or claim sparse runtime acceleration.
''', encoding='utf-8')


def execute(job_id, c, shared=None):
    folder = RUNS / job_id
    try:
        write_project(c, folder, shared)
        with LOCK:
            JOBS[job_id].update(status='running', stage='실제 PyTorch 학습 및 평가 중')
        proc = subprocess.run([sys.executable, '-u', str(folder / 'train.py')], cwd=folder,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600,
                              env={**os.environ, 'OMP_NUM_THREADS': '2'})
        result_path = folder / 'results.json'
        result = json.loads(result_path.read_text(encoding='utf-8')) if proc.returncode == 0 and result_path.exists() else None
        matched_node = re.findall(r'node=([a-z_0-9]+)', proc.stderr)
        failure_node = matched_node[-1] if matched_node else 'runtime'
        with LOCK:
            JOBS[job_id].update(status='completed' if result else 'failed', stage='검증 완료' if result else '실행 실패',
                                log=(proc.stdout + '\n' + proc.stderr)[-20000:], result=result,
                                error=None if result else {'node': failure_node, 'message': '생성 코드 실행에 실패했습니다. 로그를 확인하세요.'})
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(status='failed', stage='실행 실패', error={'node': 'runtime', 'message': str(exc)})


def busy():
    return any(j['status'] in ['queued', 'running'] for j in JOBS.values()) or any(b['status'] in ['queued', 'running'] for b in BATCHES.values())


def comparison_configs(raw):
    if not isinstance(raw, dict):
        raise ValueError('비교 설정이 필요합니다.')
    base = normalize(raw.get('base'))
    variants = raw.get('variants')
    if not isinstance(variants, list) or not 2 <= len(variants) <= 8:
        raise ValueError('비교 모델을 2~8개 선택하세요.')
    configs = []
    for i, variant in enumerate(variants):
        if not isinstance(variant, dict) or set(variant) - {'name', 'blocks', 'head', 'pool'}:
            raise ValueError('모델 조합 외 데이터·학습 설정은 비교 중 변경할 수 없습니다.')
        c = normalize({**base, **variant, 'metrics': ['f1_macro', 'recall', 'latency']})
        if not inspect(c)['valid']:
            raise ValueError(f'{c["name"]}: Mean pooling을 연결하세요.')
        configs.append(c)
    if len({c['name'] for c in configs}) != len(configs):
        raise ValueError('실험 이름을 서로 다르게 지정하세요.')
    return base, configs


def save_batch(batch):
    COMPARISONS.mkdir(parents=True, exist_ok=True)
    target = COMPARISONS / (batch['id'] + '.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(target)


def load_batches():
    if not COMPARISONS.exists():
        return
    for path in COMPARISONS.glob('*.json'):
        try:
            b = json.loads(path.read_text(encoding='utf-8'))
            if b['status'] in ['queued', 'running']:
                b['status'] = 'interrupted'
                b['error'] = '서버가 재시작되어 비교 실행이 중단되었습니다. 완료된 결과는 유지됩니다.'
                for j in b['jobs']:
                    if j['status'] in ['queued', 'running']:
                        j.update(status='interrupted', error={'message': '서버 재시작으로 중단됨'})
                save_batch(b)
            BATCHES[b['id']] = b
        except (ValueError, KeyError, OSError):
            continue


def execute_batch(batch_id):
    batch = BATCHES[batch_id]
    try:
        base = batch['base']
        shared = None
        if base['data_source'] == 'csv':
            shared = COMPARISONS / batch_id / 'shared'
            write_project(base, shared)
            fingerprint = hashlib.sha256((shared / 'flows.npz').read_bytes()).hexdigest()
            report = json.loads((shared / 'dataset_report.json').read_text(encoding='utf-8'))
        else:
            fingerprint = hashlib.sha256(json.dumps({k: base[k] for k in ['seed', 'features', 'sequence']}, sort_keys=True).encode() + (ROOT / 'pipeline_template.py').read_bytes()).hexdigest()
            report = {'name': '합성 데이터 · 실행 점검용', 'split_mode': 'fixed synthetic 360 / 120 / 180', 'features': base['features']}
        with LOCK:
            batch.update(status='running', fingerprint=fingerprint, dataset=report)
            save_batch(batch)
        for entry in batch['jobs']:
            with LOCK:
                JOBS[entry['id']] = entry
            execute(entry['id'], entry['config'], shared)
            with LOCK:
                entry.update(JOBS[entry['id']])
                save_batch(batch)
        with LOCK:
            completed = sum(j['status'] == 'completed' for j in batch['jobs'])
            batch.update(status='completed' if completed == len(batch['jobs']) else 'partial' if completed else 'failed', finished=time.time())
            save_batch(batch)
    except Exception as exc:
        with LOCK:
            batch.update(status='failed', error=str(exc), finished=time.time())
            for entry in batch['jobs']:
                if entry['status'] in ['queued', 'running']:
                    entry.update(status='failed', error={'node': 'input', 'message': str(exc)})
            save_batch(batch)


class Handler(BaseHTTPRequestHandler):
    def send(self, status, payload, mime='application/json; charset=utf-8', extra=None):
        body = json.dumps(payload, ensure_ascii=False).encode() if mime.startswith('application/json') else payload
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/dataset-ui.js':
            return self.send(200, (ROOT / 'dataset-ui.js').read_bytes(), 'text/javascript; charset=utf-8')
        if self.path == '/compare-ui.js':
            return self.send(200, (ROOT / 'compare-ui.js').read_bytes(), 'text/javascript; charset=utf-8')
        if self.path == '/api/comparisons':
            with LOCK:
                summaries = [{'id': b['id'], 'status': b['status'], 'started': b['started'], 'name': b['base']['name'], 'count': len(b['jobs'])} for b in sorted(BATCHES.values(), key=lambda b: b['started'], reverse=True)]
            return self.send(200, summaries)
        if self.path.startswith('/api/comparisons/'):
            with LOCK:
                b = copy.deepcopy(BATCHES.get(self.path.split('/')[-1]))
            return self.send(200 if b else 404, b or {'error': '비교 기록을 찾을 수 없습니다.'})
        if self.path == '/api/default':
            return self.send(200, DEFAULT)
        if self.path.startswith('/api/datasets/'):
            try:
                return self.send(200, get_dataset(self.path.split('/')[-1]))
            except ValueError as exc:
                return self.send(404, {'error': str(exc)})
        if self.path.startswith('/api/jobs/'):
            with LOCK:
                job = JOBS.get(self.path.split('/')[-1])
                payload = dict(job) if job else {'error': '실행을 찾을 수 없습니다.'}
            return self.send(200 if job else 404, payload)
        return self.send(404, {'error': 'Not found'})

    def do_POST(self):
        # Local-only server; reject cross-origin browser requests.
        origin = self.headers.get('Origin')
        if origin and origin not in [f'http://{self.headers.get("Host")}']:
            return self.send(403, {'error': '외부 출처 요청은 허용하지 않습니다.'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if self.path == '/api/upload':
                if not 0 < size <= dataset.MAX_BYTES:
                    raise ValueError('CSV는 최대 50MB까지 지원합니다.')
                content = self.rfile.read(size)
                name = unquote(self.headers.get('X-Filename', 'dataset.csv'))
                info = dataset.profile(content, name)
                dataset_id = uuid.uuid4().hex
                info['id'] = dataset_id
                UPLOADS.mkdir(exist_ok=True)
                (UPLOADS / f'{dataset_id}.csv').write_bytes(content)
                (UPLOADS / f'{dataset_id}.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
                return self.send(201, info)
            if not 0 < size <= 20000:
                raise ValueError('요청 크기가 유효하지 않습니다.')
            raw = json.loads(self.rfile.read(size))
            if self.path == '/api/compare':
                base, configs = comparison_configs(raw)
                with LOCK:
                    if busy():
                        return self.send(409, {'error': '진행 중인 실험이 있습니다. 완료 후 비교를 실행하세요.'})
                    batch_id = uuid.uuid4().hex[:12]
                    batch = {'id': batch_id, 'status': 'queued', 'base': base, 'started': time.time(),
                             'jobs': [{'id': uuid.uuid4().hex[:12], 'status': 'queued', 'stage': '비교 실행 대기', 'config': c} for c in configs]}
                    BATCHES[batch_id] = batch
                    save_batch(batch)
                threading.Thread(target=execute_batch, args=(batch_id,), daemon=True).start()
                return self.send(202, {'id': batch_id})
            c = normalize(raw)
            graph = inspect(c)
            if self.path == '/api/inspect':
                return self.send(200, {**graph, 'source': generate(c)})
            if self.path not in ['/api/run', '/api/export']:
                return self.send(404, {'error': 'Not found'})
            if not graph['valid']:
                return self.send(422, {'error': '구조도 연결을 수정하세요.', **graph})
            if self.path == '/api/export':
                buffer = io.BytesIO()
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    folder = Path(temp)
                    write_project(c, folder)
                    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                        for file in folder.iterdir():
                            archive.write(file, file.name)
                return self.send(200, buffer.getvalue(), 'application/zip', {'Content-Disposition': 'attachment; filename="sdn-ids-project.zip"'})
            with LOCK:
                if busy():
                    return self.send(409, {'error': '진행 중인 실행이 있습니다. 완료 후 다시 실행하세요.'})
                job_id = uuid.uuid4().hex[:12]
                JOBS[job_id] = {'id': job_id, 'status': 'queued', 'stage': '프로젝트 생성 중', 'config': c, 'started': time.time()}
            threading.Thread(target=execute, args=(job_id, c), daemon=True).start()
            return self.send(202, {'id': job_id})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.send(400, {'error': str(exc)})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8787)
    args = parser.parse_args()
    load_batches()
    print(f'SDN IDS Studio: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
