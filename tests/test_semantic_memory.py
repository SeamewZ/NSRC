import unittest
import numpy as np,pandas as pd
from reproduction.semantic_online import key,score,simulate
class SemanticTests(unittest.TestCase):
 def test_description_and_neutral_ablation(self):
  a=key(dict(c="Tweet",d="Bullish",z="Tesla supports DOGE payments"))
  b=key(dict(c="Tweet",d="Bullish",z="A market forecast"))
  self.assertGreater(score(a,a),score(a,b))
  self.assertEqual(score(a,a,"z"),score(a,b,"z"))
  self.assertEqual(a,key(dict(c=" tweet ",d="bullish",z="DOGE payments: Tesla supports!")))
 def fixture(self):
  n=12;times=pd.date_range("2025-01-01",periods=n,freq="2D",tz="UTC")
  idx=pd.DataFrame(dict(timestamp=times.astype(str),target_end=(times+pd.Timedelta(hours=24)).astype(str),
    asset=["BTC"]*n,memory_eligible=[True]*n,events=[1]*(n-1)+[0]))
  events=[[dict(c="Tweet",d="Bullish",z="DOGE payments")] for _ in range(n-1)]+[[]]
  return idx,np.full((n,24),1.02),np.ones((n,24)),events,np.zeros((n,10)),np.full(n,.01),dict(test_start="2025-01-09")
 def test_maturity_future_and_empty_event(self):
  args=self.fixture();p,tr=simulate(*args,changes=dict(calibrate=False))
  self.assertTrue(np.array_equal(p[0],args[2][0]))
  self.assertGreater(p[1,0],1.)
  self.assertTrue(np.array_equal(p[-1],args[2][-1]))
  changed=args[1].copy();changed[7:]*=5
  pp,_=simulate(args[0],changed,*args[2:],changes=dict(calibrate=False))
  np.testing.assert_array_equal(p[:8],pp[:8])
  self.assertTrue(all(pd.Timestamp(r["latest_memory_target"])<pd.Timestamp(r["origin"]) for r in tr if r["latest_memory_target"]))
 def test_no_match_fallback(self):
  args=list(self.fixture());args[3][1]=[dict(c="CPI",d="Below",z="inflation")]
  p,_=simulate(*args,changes=dict(calibrate=False))
  zero,_=simulate(*args,changes=dict(calibrate=False,fallback=False))
  self.assertGreater(p[1,0],1.);self.assertEqual(zero[1,0],1.)
if __name__=="__main__":unittest.main()
