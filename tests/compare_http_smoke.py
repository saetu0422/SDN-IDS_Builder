"""End-to-end comparison with an artificial uploaded CSV."""
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import Request,urlopen

BASE='http://127.0.0.1:8787'
ROOT=Path(__file__).resolve().parents[1]
def api(path,data=None,headers=None):
    body=data if isinstance(data,bytes) else json.dumps(data).encode() if data is not None else None
    with urlopen(Request(BASE+path,data=body,headers=headers or {'Content-Type':'application/json'}),timeout=30) as r:return json.load(r)

info=api('/api/upload',(ROOT/'examples/upload-demo.csv').read_bytes(),{'Content-Type':'text/csv','X-Filename':'comparison-artificial-demo.csv'})
c=api('/api/default');c.update(data_source='csv',dataset_id=info['id'],feature_columns=['bytes','packets'],label_column='label',split_mode='group',group_column='session',epochs=1)
variants=[{'name':name,'blocks':blocks,'head':head,'pool':True} for name,blocks,head in [('CNN',['conv'],'dense'),('LSTM',['lstm'],'dense'),('Attention',['attention'],'dense'),('CNN + Attention + MoE',['conv','attention'],'moe')]]
batch_id=api('/api/compare',{'base':c,'variants':variants})['id']
deadline=time.monotonic()+180
while time.monotonic()<deadline:
    b=api('/api/comparisons/'+batch_id)
    if b['status'] not in ['queued','running']:break
    time.sleep(1)
assert b['status']=='completed',b
hashes=[]
splits=[]
for j in b['jobs']:
    folder=ROOT/'runs'/j['id'];hashes.append(hashlib.sha256((folder/'flows.npz').read_bytes()).hexdigest())
    splits.append((folder/'split_rows.json').read_bytes())
    assert set(j['result']['recall'])=={'attack','normal'}
    assert set(j['result']['latency_ms'])=={'p50','p95','p99'}
assert len(set(hashes))==1 and hashes[0]==b['fingerprint']
assert len(set(splits))==1
assert (ROOT/'runs'/'comparisons'/(batch_id+'.json')).exists()
print(json.dumps({'comparison':batch_id,'models':len(b['jobs']),'status':b['status'],'identical_data_and_split':True}))
