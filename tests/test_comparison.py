import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app


class ComparisonTests(unittest.TestCase):
    def tearDown(self):
        app.BATCHES.clear(); app.JOBS.clear()

    def variants(self):
        return [{'name':'CNN','blocks':['conv'],'head':'dense','pool':True},
                {'name':'MoE','blocks':['attention'],'head':'moe','pool':True}]

    def test_comparison_locks_data_and_metrics(self):
        base, configs=app.comparison_configs({'base':{**app.DEFAULT,'metrics':[]},'variants':self.variants()})
        for c in configs:
            self.assertEqual(c['seed'],base['seed'])
            self.assertEqual(c['metrics'],['f1_macro','recall','latency'])
        for variants in [[self.variants()[0]], [self.variants()[0]]*9,
                         [{**self.variants()[0],'seed':99},self.variants()[1]],
                         [{**self.variants()[0],'pool':False},self.variants()[1]],
                         [self.variants()[0],self.variants()[0]]]:
            with self.assertRaises(ValueError):
                app.comparison_configs({'base':app.DEFAULT,'variants':variants})

    def test_failed_model_does_not_stop_batch_and_restore(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(app,'COMPARISONS',Path(temp)):
            base,configs=app.comparison_configs({'base':app.DEFAULT,'variants':self.variants()})
            batch={'id':'a'*12,'base':base,'status':'queued','started':1,
                   'jobs':[{'id':str(i),'config':c,'status':'queued'} for i,c in enumerate(configs)]}
            app.BATCHES[batch['id']]=batch
            def fake_run(id,c,shared):
                app.JOBS[id].update(status='failed' if id=='0' else 'completed',result={'f1_macro':.7} if id=='1' else None)
            with patch.object(app,'execute',side_effect=fake_run):app.execute_batch(batch['id'])
            self.assertEqual(batch['status'],'partial')
            self.assertEqual([j['status'] for j in batch['jobs']],['failed','completed'])
            app.BATCHES.clear();app.load_batches()
            self.assertEqual(app.BATCHES[batch['id']]['status'],'partial')
            b=app.BATCHES[batch['id']];b['status']='running';b['jobs'][0]['status']='running';app.save_batch(b)
            app.BATCHES.clear();app.load_batches()
            self.assertEqual(app.BATCHES[b['id']]['status'],'interrupted')
            self.assertEqual(app.BATCHES[b['id']]['jobs'][1]['status'],'completed')

    def test_batch_reservation_blocks_other_runs(self):
        app.BATCHES['x']={'status':'queued'}
        self.assertTrue(app.busy())
        app.BATCHES['x']['status']='completed'
        self.assertFalse(app.busy())
