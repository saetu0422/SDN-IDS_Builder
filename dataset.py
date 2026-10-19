"""CSV ingestion and reproducible train/validation/test splitting."""
import csv
import hashlib
import io
import math
from collections import Counter

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 200000


def parse_csv(content):
    if len(content) > MAX_BYTES:
        raise ValueError('CSV는 최대 50MB까지 지원합니다.')
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = content.decode('cp949')
        except UnicodeDecodeError as exc:
            raise ValueError('UTF-8 또는 CP949 CSV 파일을 사용하세요.') from exc
    try:
        dialect = csv.Sniffer().sniff(text[:10000], delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text, newline=''), dialect, strict=True)
    try:
        columns = [v.strip() for v in next(reader)]
        if not 2 <= len(columns) <= 128 or any(not c or len(c) > 160 for c in columns) or len(set(columns)) != len(columns):
            raise ValueError('헤더는 중복 없는 2~128개 열이어야 합니다 (열 이름 최대 160자).')
        rows = []
        for line, row in enumerate(reader, 2):
            if not row or all(not v.strip() for v in row):
                continue
            if len(row) != len(columns):
                raise ValueError(f'CSV {line}행: 열 개수가 헤더와 다릅니다.')
            rows.append([v.strip() for v in row])
            if len(rows) > MAX_ROWS:
                raise ValueError('MVP는 최대 200,000행까지 지원합니다.')
    except (csv.Error, StopIteration) as exc:
        raise ValueError('CSV 형식이나 헤더를 확인하세요.') from exc
    if not rows:
        raise ValueError('데이터 행이 없습니다.')
    return columns, rows


def finite_number(value):
    try:
        return bool(value) and math.isfinite(float(value)) and abs(float(value)) <= 1e30
    except ValueError:
        return False


def profile(content, name):
    columns, rows = parse_csv(content)
    fields = []
    for i, col in enumerate(columns):
        values = [r[i] for r in rows]
        counts = Counter(values)
        numeric = sum(finite_number(v) for v in values)
        fields.append({'name': col, 'numeric': numeric == len(rows), 'invalid_numeric': len(rows) - numeric,
                       'unique': len(counts), 'missing': counts.get('', 0),
                       'counts': dict(counts) if len(counts) <= 50 else None})
    return {'name': name[:160], 'rows': len(rows), 'columns': fields,
            'preview': rows[:5], 'sha256': hashlib.sha256(content).hexdigest()}


def validate_selection(info, settings):
    fields = {c['name']: c for c in info['columns']}
    selected = settings.get('feature_columns')
    label = settings.get('label_column')
    if not isinstance(selected, list) or not 1 <= len(selected) <= 64 or any(not isinstance(c, str) for c in selected) or len(set(selected)) != len(selected):
        raise ValueError('숫자형 특성을 1~64개 선택하세요.')
    if label not in fields or label in selected:
        raise ValueError('라벨 열을 선택하고 입력 특성에서 제외하세요.')
    if any(c not in fields or not fields[c]['numeric'] for c in selected):
        raise ValueError('선택한 특성에 결측값, 무한대 또는 숫자가 아닌 값이 있습니다. CSV를 정리하세요.')
    field = fields[label]
    if field['missing'] or not 2 <= field['unique'] <= 50:
        raise ValueError('라벨은 결측값 없이 2~50개 클래스여야 합니다.')
    if min(field['counts'].values()) < 7:
        raise ValueError('각 클래스는 최소 7행이 필요합니다. 클래스 분포를 확인하세요.')
    mode = settings.get('split_mode', 'stratified')
    if mode not in ['stratified', 'group']:
        raise ValueError('지원하지 않는 데이터 분할 방식입니다.')
    if mode == 'group':
        group = settings.get('group_column')
        if group not in fields or group == label or group in selected or fields[group]['missing'] or fields[group]['unique'] < 3:
            raise ValueError('그룹 열은 특성·라벨과 달라야 하며, 결측값 없이 최소 3개 그룹이 필요합니다.')
    return sorted(field['counts'])


def prepare(content, info, settings, seed):
    import numpy as np
    classes = validate_selection(info, settings)
    columns, rows = parse_csv(content)
    indices = [columns.index(c) for c in settings['feature_columns']]
    label_idx = columns.index(settings['label_column'])
    mapping = {v: i for i, v in enumerate(classes)}
    x = np.asarray([[float(row[i]) for i in indices] for row in rows], dtype=np.float32)[:, None, :]
    y = np.asarray([mapping[row[label_idx]] for row in rows], dtype=np.int64)
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    mode = settings.get('split_mode', 'stratified')
    if mode == 'stratified':
        for cls in range(len(classes)):
            ids = rng.permutation(np.flatnonzero(y == cls))
            val_n = max(1, int(len(ids) * .15))
            test_n = max(1, int(len(ids) * .15))
            for target, chunk in zip(splits, [ids[val_n + test_n:], ids[:val_n], ids[val_n:val_n + test_n]]):
                target.extend(chunk.tolist())
    else:
        group_idx = columns.index(settings['group_column'])
        groups = sorted(set(row[group_idx] for row in rows))
        groups = rng.permutation(groups).tolist()
        val_n = max(1, int(len(groups) * .15))
        test_n = max(1, int(len(groups) * .15))
        group_split = {g: s for s, chunk in enumerate([groups[val_n + test_n:], groups[:val_n], groups[val_n:val_n + test_n]]) for g in chunk}
        for i, row in enumerate(rows):
            splits[group_split[row[group_idx]]].append(i)
    arrays, distributions, row_ids = {}, {}, {}
    for name, ids in zip(['train', 'val', 'test'], splits):
        ids = rng.permutation(ids).astype(np.int64)
        if len(ids) == 0 or len(np.unique(y[ids])) != len(classes):
            raise ValueError('분할 후 일부 클래스가 없습니다. 그룹 구성이나 데이터 수를 확인하세요. 분할 방식을 자동 변경하지 않습니다.')
        arrays['X_' + name], arrays['y_' + name] = x[ids], y[ids]
        row_ids[name] = (ids + 2).tolist()
        distributions[name] = {c: int((y[ids] == i).sum()) for i, c in enumerate(classes)}
    report = {'name': info['name'], 'sha256': info['sha256'], 'rows': len(rows), 'features': settings['feature_columns'],
              'label': settings['label_column'], 'classes': classes, 'class_mapping': mapping,
              'split_mode': mode, 'group_column': settings.get('group_column') if mode == 'group' else None,
              'split_target': '70/15/15 per class' if mode == 'stratified' else '70/15/15 of groups; row ratios may differ',
              'distribution': distributions, 'sequence': 1, 'seed': seed,
              'preprocessing': 'Reject missing/non-numeric features; scaler fitted on training data only',
              'row_reference': 'split_rows.json uses parsed record numbers including header (starts at 2), omitting blank lines; not physical CSV line numbers',
              'warning': 'Random row splits can share sessions; use a session/group column to isolate related flows.' if mode == 'stratified' else 'Group-disjoint split. Temporal generalization is not measured.'}
    return arrays, report, row_ids
