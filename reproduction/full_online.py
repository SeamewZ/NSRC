"""Causal ARM90 and prespecified ablations on generic frozen backbone forecasts."""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path("runs/full_evaluation_20260921")
CONFIG=dict(window=90,half_life=60.,pool=.25,tau=10.,normalize=True,representation="agnostic",calibrate=True,online=True,gain_shrink=10.,event_memory=True)
VARIANTS={"ARM90":{},"NSRC90":dict(representation="semantic"),"w_o_calibration":dict(calibrate=False),"w_o_online_update":dict(online=False),"w_o_recency":dict(half_life=np.inf),"w_o_volatility":dict(normalize=False),"w_o_sharing":dict(pool=0.),"w_o_support_shrinkage":dict(tau=0.,gain_shrink=0.),"window30":dict(window=30),"window180":dict(window=180),"topic90":dict(representation="topic"),"context90":dict(representation="context"),"w_o_event_memory_filter":dict(event_memory=False)}
STOPWORDS={"the","a","an","and","or","of","to","in","on","for","with","from","by","is","are","was","were","at","as","this","that","it","its"}
def semantic_tokens(text):
    words=re.findall(r"[a-z0-9]+",str(text).lower())
    return set(w for w in words if len(w)>1 and w not in STOPWORDS)
def semantic_similarity(query, memory):
    """Similarity for the symbolic key (event class, direction, description)."""
    qc=str(query.get("c","")).strip().lower(); mc=str(memory.get("c","")).strip().lower()
    qd=str(query.get("d","")).strip().lower(); md=str(memory.get("d","")).strip().lower()
    qt=semantic_tokens(query.get("z",query.get("text",""))); mt=semantic_tokens(memory.get("z",memory.get("text","")))
    union=qt|mt; jaccard=(len(qt&mt)/len(union)) if union else 0.0
    return 0.50*float(qc==mc)+0.25*float(qd==md)+0.25*jaccard

def gain(delta,effect,assets,shrink=10.):
    n=len(delta)
    if n<3:return 0.
    _,inv,counts=np.unique(assets,return_inverse=True,return_counts=True)
    w=np.repeat(1/counts[inv]/len(counts)/24,24);e=effect.ravel();d=delta.ravel();keep=np.abs(e)>1e-14
    if not keep.any():return 0.
    v=d[keep]/e[keep];ww=w[keep]*np.abs(e[keep]);order=np.argsort(v,kind="stable")
    g1=v[order[np.searchsorted(np.cumsum(ww[order]),ww.sum()/2)]];g2=np.sum(w*e*d)/np.sum(w*e*e)
    if g1*g2<=0:return 0.
    return float(max(0.,min(g1,g2,1.5))*n/(n+shrink))
def simulate(idx,y,base,events,x,sigma,manifest,changes=None,stop=None):
    cfg=dict(CONFIG,**(changes or {}));ts=pd.to_datetime(idx.timestamp,utc=True).astype("int64").to_numpy()/86400e9;end=pd.to_datetime(idx.target_end,utc=True).astype("int64").to_numpy()/86400e9
    test_start=pd.Timestamp(manifest["test_start"],tz="UTC").value/86400e9
    order=np.flatnonzero(idx.memory_eligible.to_numpy()&(idx.events.to_numpy()>0));order=order[np.argsort(ts[order],kind="stable")]
    if stop is not None:order=order[ts[order]<stop]
    assets=idx.asset.to_numpy();pred=np.array(base,dtype=float,copy=True);effects=np.zeros_like(pred);root=np.sqrt(np.arange(1,25));scale=sigma[:,None]*root
    delta=y-base;values=np.clip(delta/scale,-5,5) if cfg["normalize"] else np.clip(delta,-.2,.2)
    trace=[];processed=[];frozen_gain=None
    for q in order:
        past=np.asarray(processed,dtype=int) if cfg["event_memory"] else np.flatnonzero(idx.memory_eligible.to_numpy())
        past=past[(end[past]<ts[q])&(ts[past]>=ts[q]-cfg["window"])]
        if not cfg["online"] and ts[q]>=test_start:past=past[end[past]<test_start]
        if len(past):
            assert end[past].max()<ts[q]
            weights=np.where(assets[past]==assets[q],1.,cfg["pool"])*np.exp2(-(ts[q]-ts[past])/cfg["half_life"])
            if cfg["representation"]=="context":weights*=np.exp(-np.mean((x[past]-x[q])**2,axis=1)/2)
            answers=[]
            for e in events[q]:
                sim=np.ones(len(past))
                if cfg["representation"]=="semantic":
                    semantic_scores=[]
                    for j in past:
                        semantic_scores.append(max((semantic_similarity(e,m) for m in events[j]), default=0.0))
                    semantic_scores=np.asarray(semantic_scores,dtype=float)
                    if np.any(semantic_scores>0): sim=semantic_scores
                elif cfg["representation"]!="agnostic":
                    query_topics=set(e["topic"].split("+"));similar=[]
                    for j in past:
                        ws=[]
                        for m in events[j]:
                            t=set(m["topic"].split("+"));ws.append((.1+.9*len(t&query_topics)/max(len(t|query_topics),1)) if e["d"]==m["d"] else 0.)
                        similar.append(np.mean(ws))
                    sim=np.asarray(similar)
                w=weights*sim;support=w.sum();response=(w@values[past])/(support+cfg["tau"]) if support>0 else np.zeros(24)
                answers.append(response*scale[q] if cfg["normalize"] else response)
            effects[q]=np.mean(answers,axis=0)
        known=past[np.any(np.abs(effects[past])>1e-14,axis=1)]
        g=gain(delta[known],effects[known],assets[known],cfg["gain_shrink"]) if cfg["calibrate"] else 1.
        if not cfg["online"] and ts[q]>=test_start:
            if frozen_gain is None:frozen_gain=g
            g=frozen_gain
        pred[q]+=g*effects[q]
        trace.append(dict(row=int(q),asset=assets[q],origin=idx.iloc[q].timestamp,memory_n=len(past),calibration_n=len(known),latest_memory_target=idx.iloc[past[np.argmax(end[past])]].target_end if len(past) else None,gain=g,effect_l2=float(np.linalg.norm(effects[q]))))
        processed.append(q)
    return pred,trace

def run(dataset,backbone="Persistence",seed=0,ablate=True):
    data=ROOT/"data"/dataset;out=ROOT/"evaluation"/dataset/f"{backbone}_s{seed}";out.mkdir(parents=True,exist_ok=True)
    old_predictions=None
    if (out/"COMPLETE").exists():
        previous=np.load(out/"predictions.npz")
        if not ablate or set(VARIANTS).issubset(previous.files):return
        old_predictions={k:previous[k] for k in previous.files}
    idx=pd.read_csv(data/"index.csv");y=np.load(data/"windows.npz")["y"].astype(float);feat=np.load(data/"features.npz");events=json.loads((data/"events.json").read_text());manifest=json.loads((data/"manifest.json").read_text())
    if backbone=="Persistence":base=np.ones_like(y);provenance={"method":"Persistence","seed":None}
    else:
        source=ROOT/"baselines"/dataset/f"{backbone}_s{seed}";base=np.load(source/"predictions.npy").astype(float);provenance=json.loads((source/"provenance.json").read_text());assert base.shape==y.shape
        if dataset!="news":
            original=np.load(Path(provenance["data"])/"windows.npz");current=np.load(data/"windows.npz");assert np.array_equal(current["x"],original["x"]) and np.array_equal(current["y"],original["y"]),"Old checkpoint row alignment differs"
    assert np.isfinite(base).all()
    keep=idx.test.to_numpy();preds={"no_correction":base[keep]};traces=[]
    if old_predictions is not None:
        preds.update({k:v for k,v in old_predictions.items() if k!="y"});traces=pd.read_csv(out/"update_trace.csv").to_dict("records")
    variants=VARIANTS if ablate else {"ARM90":{}}
    for name,changes in variants.items():
        if name in preds:continue
        p,t=simulate(idx,y,base,events,feat["x"],feat["sigma"],manifest,changes)
        assert np.isfinite(p).all() and np.array_equal(p[idx.events==0],base[idx.events==0])
        preds[name]=p[keep];traces.extend(dict(variant=name,**r) for r in t)
        print(dataset,backbone,seed,name,"done",flush=True)
    # Test intervention: unseen outcomes must have no effect on earlier predictions.
    cut=pd.Timestamp(manifest["test_start"],tz="UTC").value/86400e9+14
    end=pd.to_datetime(idx.target_end,utc=True).astype("int64").to_numpy()/86400e9;ts=pd.to_datetime(idx.timestamp,utc=True).astype("int64").to_numpy()/86400e9
    changed=y.copy();changed[end>=cut]*=1.5
    pa,_=simulate(idx,y,base,events,feat["x"],feat["sigma"],manifest,stop=cut);pb,_=simulate(idx,changed,base,events,feat["x"],feat["sigma"],manifest,stop=cut)
    assert np.array_equal(pa[ts<cut],pb[ts<cut]),"Future target intervention changed earlier predictions"
    if old_predictions is not None:
        for key,value in old_predictions.items():assert np.array_equal(value,y[keep] if key=="y" else preds[key])
    np.savez_compressed(out/"predictions.tmp.npz",y=y[keep],**preds);(out/"predictions.tmp.npz").replace(out/"predictions.npz");idx[keep].to_csv(out/"test_index.csv",index=False);pd.DataFrame(traces).to_csv(out/"update_trace.csv",index=False)
    verification=dict(all_predictions_finite=True,no_event_identity=True,future_target_intervention_invariant=True,strictly_matured_memory=True,base_provenance=provenance,config=CONFIG,variants=variants,protocol_sha256=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest())
    (out/"verification.json").write_text(json.dumps(verification,indent=2));(out/"COMPLETE").write_text("Causal predictions and checks complete\n")
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--dataset",required=True);a.add_argument("--backbone",default="Persistence");a.add_argument("--seed",type=int,default=0);a.add_argument("--primary-only",action="store_true");args=a.parse_args();run(args.dataset,args.backbone,args.seed,not args.primary_only)
