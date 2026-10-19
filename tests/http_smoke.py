"""Manual integration check against the running local server."""
import io
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
import zipfile

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8787'


def request(path, value=None, headers=None):
    body = value if isinstance(value, bytes) else json.dumps(value).encode() if value is not None else None
    req = Request(BASE + path, data=body, headers=headers or {'Content-Type':'application/json'})
    with urlopen(req, timeout=30) as response:
        data=response.read()
        return data if response.headers.get_content_type()=='application/zip' else json.loads(data)


root=Path(__file__).resolve().parents[1]
info=request('/api/upload',(root/'examples/upload-demo.csv').read_bytes(),{'Content-Type':'text/csv','X-Filename':'upload-demo.csv'})
config=request('/api/default')
config.update(data_source='csv',dataset_id=info['id'],feature_columns=['bytes','packets'],label_column='label',
              split_mode='group',group_column='session',epochs=1,head='moe')
graph=request('/api/inspect',config)
assert graph['valid'] and graph['classes']==2
assert '[B, 1, 2]' == graph['nodes'][0]['shape']
archive=zipfile.ZipFile(io.BytesIO(request('/api/export',config)))
assert archive.testzip() is None
assert {'flows.npz','dataset_report.json','split_rows.json','train.py'} <= set(archive.namelist())
job=request('/api/run',config)
deadline=time.monotonic()+90
while time.monotonic()<deadline:
    result=request('/api/jobs/'+job['id'])
    if result['status'] in ['completed','failed']:break
    time.sleep(.5)
assert result['status']=='completed',result
assert result['result']['data_source']=='uploaded_csv'
assert result['result']['dataset']['split_mode']=='group'
print(json.dumps({'upload':info['id'],'job':job['id'],'classes':result['result']['classes'],
                  'samples':result['result']['samples'],'export_files':archive.namelist()},ensure_ascii=False))
