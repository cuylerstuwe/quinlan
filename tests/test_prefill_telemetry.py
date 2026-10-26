"""Deterministic phase accounting and scheduler-hook regression checks."""
from types import SimpleNamespace
import unittest
from quinlan import server as service

class PrefillTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.now=0
        self.timer=service.PrefillTelemetry(clock_ns=lambda:self.now,max_records=2)
    def req(self,n=10000,cached=9900):return SimpleNamespace(n_prompt=n,start=cached,err=None)
    def measured(self,req,seconds=.05):
        self.timer.begin(req);self.now+=int(seconds*1e9);self.timer.finish(req)
        return self.timer.take(req)
    def test_counts_only_processed_tokens(self):
        t=self.measured(self.req())
        self.assertEqual(t,{"prompt_n":100,"prompt_ms":50,"prompt_per_second":2000,"cache_n":9900,"scope":"server_prefill_wall"})
    def test_cold_prefill(self):
        t=self.measured(self.req(1000,0),.5)
        self.assertEqual(t['prompt_n'],1000);self.assertEqual(t['prompt_per_second'],2000)
    def test_queue_wait_is_excluded(self):
        req=self.req();self.now+=100_000_000_000
        t=self.measured(req,.5)
        self.assertEqual(t['prompt_ms'],500)
    def test_invalid_failed_or_zero_work_is_omitted(self):
        for n,cached in [(0,0),(100,100),(100,101),(100,-1)]:
            self.assertIsNone(self.measured(self.req(n,cached)))
        req=self.req();req.err='failed';self.assertIsNone(self.measured(req))
        self.assertIsNone(self.measured(self.req(),0))
        req=self.req();self.timer.begin(req);self.timer.finish(req);self.assertIsNone(self.timer.take(req))
    def test_missing_incomplete_and_discarded(self):
        req=self.req();self.assertIsNone(self.timer.take(req));self.timer.begin(req);self.assertIsNone(self.timer.take(req))
        self.timer.begin(req);self.timer.discard(req);self.now+=100;self.timer.finish(req);self.assertIsNone(self.timer.take(req))
    def test_changed_counts_are_omitted(self):
        req=self.req();self.timer.begin(req);req.start=0;self.now+=1_000_000;self.timer.finish(req)
        self.assertIsNone(self.timer.take(req))
    def test_bounded_numeric_records_and_consumption(self):
        refs=[self.req() for _ in range(3)]
        for req in refs:self.timer.begin(req)
        self.assertEqual(len(self.timer.records),2);self.assertIsNone(self.timer.take(refs[0]))
        self.now+=1_000_000;self.timer.finish(refs[-1]);self.assertIsNotNone(self.timer.take(refs[-1]));self.assertIsNone(self.timer.take(refs[-1]))
    def test_scheduler_hooks_preserve_arguments_and_cover_both_paths(self):
        calls=[]
        class Base:
            def __init__(self):pass
            def _prefill_start(self,req,slot):calls.append(('long',req,slot))
            def _prefill_wave(self,wave):calls.append(('short',wave))
            def _activate(self,req,slot,seed):calls.append(('activate',req,slot,seed))
        class Engine(service.PrefillTimingMixin,Base):pass
        eng=Engine();eng.prefill_telemetry=self.timer
        req=self.req();eng._prefill_start(req,2);self.now+=50_000_000;eng._activate(req,2,73)
        self.assertEqual(self.timer.take(req)['prompt_ms'],50)
        self.assertEqual(calls[:2],[('long',req,2),('activate',req,2,73)])
        req2=self.req(50,0);wave=[(req2,1)];eng._prefill_wave(wave);self.now+=10_000_000;eng._activate(req2,1,12)
        self.assertEqual(self.timer.take(req2)['prompt_ms'],10)
        self.assertEqual(calls[2:],[('short',wave),('activate',req2,1,12)])

if __name__=='__main__':unittest.main()
