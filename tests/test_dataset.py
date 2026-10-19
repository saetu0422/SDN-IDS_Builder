import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import dataset


def fixture():
    lines = ['session,bytes,packets,label']
    for group in range(20):
        for label in ['normal', 'attack']:
            for j in range(3):
                lines.append(f's{group},{10+group+j+(100 if label=="attack" else 0)},{j+1},{label}')
    return ('\n'.join(lines)+'\n').encode()


class DatasetTests(unittest.TestCase):
    def settings(self, mode='stratified'):
        return dict(feature_columns=['bytes','packets'], label_column='label', split_mode=mode, group_column='session')

    def test_stratified_reproducible_disjoint(self):
        content=fixture(); info=dataset.profile(content,'fixture.csv')
        a, report, rows=dataset.prepare(content,info,self.settings(),42)
        b, _, same=dataset.prepare(content,info,self.settings(),42)
        self.assertEqual(rows,same)
        self.assertEqual(report['classes'],['attack','normal'])
        self.assertEqual(a['X_train'].shape[1:],(1,2))
        sets=[set(v) for v in rows.values()]
        self.assertEqual(len(set.union(*sets)),120)
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
        self.assertEqual([len(v) for v in rows.values()],[84,18,18])

    def test_group_split_no_shared_sessions(self):
        content=fixture(); info=dataset.profile(content,'fixture.csv')
        _, report, rows=dataset.prepare(content,info,self.settings('group'),42)
        _, raw=dataset.parse_csv(content)
        groups=[{raw[i-2][0] for i in ids} for ids in rows.values()]
        self.assertFalse(groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
        self.assertEqual(report['split_mode'],'group')

    def test_invalid_data_and_leakage_selection(self):
        for content in [b'a,a\n1,2',b'a,b\n1,2,3', b'a,b\n']:
            with self.assertRaises(ValueError): dataset.profile(content,'bad.csv')
        content=fixture(); info=dataset.profile(content,'fixture.csv')
        for settings in [{**self.settings(),'feature_columns':['label']}, {**self.settings(),'feature_columns':['session']}, {**self.settings('group'),'group_column':'bytes'}]:
            with self.assertRaises(ValueError): dataset.validate_selection(info,settings)
        bad=fixture().replace(b',10,',b',nan,',1)
        with self.assertRaises(ValueError): dataset.validate_selection(dataset.profile(bad,'bad.csv'),self.settings())

    def test_csv_binary_moe_export_runs_and_scaler_uses_train(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); uploads=base/'uploads';uploads.mkdir()
            content=fixture();info=dataset.profile(content,'fixture.csv');id='a'*32
            (uploads/f'{id}.csv').write_bytes(content)
            (uploads/f'{id}.json').write_text(json.dumps(info),encoding='utf-8')
            with patch.object(app,'UPLOADS',uploads):
                c=app.normalize({**app.DEFAULT,**self.settings('group'),'data_source':'csv','dataset_id':id,'head':'moe','blocks':['conv','lstm','attention'],'epochs':1})
                self.assertEqual(c['features'],2);self.assertEqual(c['sequence'],1)
                self.assertEqual(app.inspect(c)['classes'],2)
                folder=base/'export';app.write_project(c,folder)
            p=subprocess.run([sys.executable,str(folder/'train.py')],cwd=folder,capture_output=True,text=True,timeout=90)
            self.assertEqual(p.returncode,0,p.stderr)
            result=json.loads((folder/'results.json').read_text())
            self.assertEqual(result['data_source'],'uploaded_csv')
            self.assertEqual(set(result['recall']),{'attack','normal'})
            self.assertEqual(len(result['confusion']),2)
            self.assertEqual(sum(map(sum,result['confusion'])),18)
            with np.load(folder/'scaler.npz') as scaler, np.load(folder/'flows.npz') as flows:
                np.testing.assert_allclose(scaler['mean'],flows['X_train'].mean(axis=(0,1),keepdims=True))


if __name__=='__main__': unittest.main()
