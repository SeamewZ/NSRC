"""Independent, exploratory semantic-memory extension of the immutable ARM90 baseline."""
import argparse,functools,hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from reproduction.full_online import CONFIG,gain,simulate as simulate_arm
BASE=Path("runs/full_evaluation_20260921")
ROOT=Path("runs/nsrc_semantic_20260921")
STOPWORDS={"the","a","an","and","or","of","to","in","on","for","with","from","by","is","are","was","were","at","as","this","that","it","its"}
VARIANTS={
 "NSRC90":{},
 "w_o_z":{"omit":"z"},
 "w_o_c":{"omit":"c"},
 "w_o_d":{"omit":"d"},
 "w_o_fallback":{"fallback":False},
 "w_o_calibration":{"calibrate":False},
 "w_o_online_update":{"online":False},
 "w_o_recency":{"half_life":float("inf")},
 "w_o_volatility":{"normalize":False},
 "w_o_sharing":{"pool":0.},
 "w_o_support_shrinkage":{"tau":0.,"gain_shrink":0.},
 "shuffled_memory":{"shuffle":True},
}
def key(e):
 return (str(e.get("c","")).strip().lower(),str(e.get("d","")).strip().lower(),
         tuple(sorted({w for w in re.findall(r"[a-z0-9]+",str(e.get("z",e.get("text",""))).lower()) if len(w)>1 and w not in STOPWORDS})))
@functools.lru_cache(maxsize=100000)
def score(a,b,omit=None):
 c=float(a[0]==b[0]);d=float(a[1]==b[1]);u=set(a[2])|set(b[2]);z=len(set(a[2])&set(b[2]))/len(u) if u else 0.
 # An omitted component is neutral (one), preserving the score's maximum and scale.
 return .5*(1. if omit=="c" else c)+.25*(1. if omit=="d" else d)+.25*(1. if omit=="z" else z)
def simulate(idx,y,base,events,x,sigma,manifest,changes=None,stop=None):
 cfg=dict(CONFIG,fallback=True,omit=None,shuffle=False,**{})
 cfg.update(changes or {})
 ts=pd.to_datetime(idx.timestamp,utc=True).astype("int64").to_numpy()/86400e9
 end=pd.to_datetime(idx.target_end,utc=True).astype("int64").to_numpy()/86400e9
 test_start=pd.Timestamp(manifest["test_start"],tz="UTC").value/86400e9
 order=np.flatnonzero(idx.memory_eligible.to_numpy()&(idx.events.to_numpy()>0))
 order=order[np.argsort(ts[order],kind="stable")]
 if stop is not None:order=order[ts[order]<stop]
 assets=idx.asset.to_numpy();keys=[tuple(key(e) for e in es) for es in events]
 pred=base.astype(float,copy=True);effects=np.zeros_like(pred);scale=sigma[:,None]*np.sqrt(np.arange(1,25))
 delta=y-base;values=np.clip(delta/scale,-5,5) if cfg["normalize"] else np.clip(delta,-.2,.2)
 processed=[];trace=[];frozen_gain=None
 for q in order:
  past=np.asarray(processed,dtype=int)
  past=past[(end[past]<ts[q])&(ts[past]>=ts[q]-cfg["window"])]
  if not cfg["online"] and ts[q]>=test_start:past=past[end[past]<test_start]
  support=[];fallbacks=0
  if len(past):
   weights=np.where(assets[past]==assets[q],1.,cfg["pool"])*np.exp2(-(ts[q]-ts[past])/cfg["half_life"])
   assigned=past.copy()
   if cfg["shuffle"]:
    rng=np.random.default_rng(20260921+int(q))
    for a in np.unique(assets[past]):
     pos=np.flatnonzero(assets[past]==a);assigned[pos]=rng.permutation(past[pos])
   answers=[]
   for e in keys[q]:
    sim=np.asarray([max((score(e,m,cfg["omit"]) for m in keys[j]),default=0.) for j in assigned])
    if not np.any(sim>0) and cfg["fallback"]:sim=np.ones(len(past));fallbacks+=1
    w=weights*sim;n=w.sum();support.append(float(n))
    response=(w@values[past])/(n+cfg["tau"]) if n>0 else np.zeros(24)
    answers.append(response*scale[q] if cfg["normalize"] else response)
   effects[q]=np.mean(answers,axis=0)
  known=past[np.any(abs(effects[past])>1e-14,axis=1)]
  g=gain(delta[known],effects[known],assets[known],cfg["gain_shrink"]) if cfg["calibrate"] else 1.
  if not cfg["online"] and ts[q]>=test_start:
   if frozen_gain is None:frozen_gain=g
   g=frozen_gain
  pred[q]+=g*effects[q]
  trace.append(dict(row=int(q),asset=assets[q],origin=idx.iloc[q].timestamp,memory_n=len(past),
   calibration_n=len(known),latest_memory_target=idx.iloc[past[np.argmax(end[past])]].target_end if len(past) else None,
   gain=g,effect_l2=float(np.linalg.norm(effects[q])),semantic_support=float(np.mean(support)) if support else 0.,
   fallback_events=fallbacks,query_events=len(keys[q])))
  processed.append(q)
 return pred,trace
def run(ds,backbone,seed,ablate=False):
 out=ROOT/"evaluation"/ds/f"{backbone}_s{seed}";out.mkdir(parents=True,exist_ok=True)
 if (out/"COMPLETE").exists():return
 data=BASE/"data"/ds;idx=pd.read_csv(data/"index.csv")
 y=np.load(data/"windows.npz")["y"].astype(float);features=np.load(data/"features.npz")
 events=json.loads((data/"events.json").read_text());manifest=json.loads((data/"manifest.json").read_text())
 base=np.ones_like(y) if backbone=="Persistence" else np.load(BASE/"baselines"/ds/f"{backbone}_s{seed}"/"predictions.npy").astype(float)
 test=idx.test.to_numpy();archive=BASE/"evaluation"/ds/f"{backbone}_s{seed}"
 old=np.load(archive/"predictions.npz")
 np.testing.assert_array_equal(old["y"],y[test]);np.testing.assert_array_equal(old["no_correction"],base[test])
 predictions={"no_correction":base[test],"ARM90":old["ARM90"]};traces=[];checks=[]
 cut=pd.Timestamp(manifest["test_start"],tz="UTC").value/86400e9+14
 end=pd.to_datetime(idx.target_end,utc=True).astype("int64").to_numpy()/86400e9
 times=pd.to_datetime(idx.timestamp,utc=True).astype("int64").to_numpy()/86400e9
 changed=y.copy();changed[end>=cut]*=1.5
 for variant,changes in (VARIANTS if ablate else {"NSRC90":{}}).items():
  p,t=simulate(idx,y,base,events,features["x"],features["sigma"],manifest,changes)
  p2,_=simulate(idx,changed,base,events,features["x"],features["sigma"],manifest,changes,stop=cut)
  np.testing.assert_array_equal(p[times<cut],p2[times<cut])
  assert np.isfinite(p).all();np.testing.assert_array_equal(p[idx.events==0],base[idx.events==0])
  for row in t:
   if row["latest_memory_target"] is not None:assert pd.Timestamp(row["latest_memory_target"])<pd.Timestamp(row["origin"])
  predictions[variant]=p[test];traces.extend(dict(variant=variant,**row) for row in t)
  checks.append(dict(variant=variant,future_target_invariance=True,no_event_identity=True,finite=True))
  print(ds,backbone,seed,variant,"verified",flush=True)
 pilot=ROOT/"pilot/evaluation"/ds/f"{backbone}_s{seed}"/"predictions.npz"
 if pilot.exists():
  np.testing.assert_allclose(predictions["NSRC90"],np.load(pilot)["NSRC90"],rtol=1e-12,atol=1e-14)
 np.savez_compressed(out/"predictions.npz",y=y[test],**predictions);idx[test].to_csv(out/"test_index.csv",index=False)
 pd.DataFrame(traces).to_csv(out/"update_trace.csv",index=False)
 (out/"verification.json").write_text(json.dumps(dict(checks=checks,baseline_predictions_sha256=hashlib.sha256((archive/"predictions.npz").read_bytes()).hexdigest(),config=CONFIG,
   protocol_sha256=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest(),
   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),exploratory=True),indent=2))
 (out/"COMPLETE").touch()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--all",action="store_true");ap.add_argument("--dataset");ap.add_argument("--backbone",default="DLinear");ap.add_argument("--seed",type=int,default=2026);args=ap.parse_args()
 if args.all:
  for flag in sorted((BASE/"evaluation").glob("*/*/COMPLETE")):
   ds=flag.parent.parent.name;b,s=flag.parent.name.rsplit("_s",1);run(ds,b,int(s),b=="DLinear")
  (ROOT/"MEASUREMENTS_COMPLETE").touch()
 else:run(args.dataset,args.backbone,args.seed,args.backbone=="DLinear")
if __name__=="__main__":main()
