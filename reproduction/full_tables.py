"""Separate main, ablation and stability summaries from one verified set of measurements."""
import pandas as pd,numpy as np
from reproduction.full_report import ROOT,R
f=pd.read_csv(R/"seed_summary.csv");f=f[(f.subset=="event")&(f.horizon=="avg24")].copy();baselines=["Persistence","DLinear","GPT4TS","PatchTST","iTransformer","TimesNet","Time-LLM","CAMEF"]
main_names=baselines+["ARM90"]+[x+"+ARM90" for x in baselines if x!="Persistence"];f[f.method.isin(main_names)].to_csv(R/"main_comparison.csv",index=False)
rows=[]
for base in baselines:
 full="ARM90" if base=="Persistence" else base+"+ARM90"
 for ds in f.dataset.unique():
  q=f[f.dataset==ds]
  if not (q.method==full).any():continue
  names=[base,full]
  if base=="Persistence":names+=list(q[q.method.str.startswith("w_o_")|q.method.isin(["window30","window180","topic90","context90"])].method.unique())
  else:names+=list(q[q.method.str.startswith(base+"+")&(q.method!=full)].method.unique())
  for _,r in q[q.method.isin(names)].iterrows():
   row=r.to_dict();row["backbone"]=base;row["variant"]="no correction" if r.method==base else ("full ARM90" if r.method==full else r.method.removeprefix(base+"+").replace("w_o_","w/o ").replace("_"," "));rows.append(row)
pd.DataFrame(rows).to_csv(R/"ablation_table.csv",index=False)
stability=[]
for file,keys in [("temporal_stability.csv",["scale","period"]),("horizon_stability.csv",["horizon"]),("volatility_stability.csv",["volatility"])]:
 data=pd.read_csv(R/file);join=["dataset","asset","subset","seed"]+keys
 for base in baselines:
  full="ARM90" if base=="Persistence" else base+"+ARM90";a=data[data.method==full];b=data[data.method==base];pair=a.merge(b,on=join,suffixes=("_full","_base"))
  if pair.empty:continue
  for metric in ["mse","mae","mape"]:
   pair[metric+"_improvement_pct"]=100*(1-pair[metric+"_full"]/pair[metric+"_base"].clip(lower=1e-16))
  pair["reference"]=base;pair.to_csv(R/(file.replace(".csv","")+"_"+base+"_improvement.csv"),index=False)
  group=["dataset","asset","subset","seed"]+(["scale"] if "scale" in keys else [])
  for values,g in pair.groupby(group):
   record=dict(zip(group,values));record.update(reference=base,analysis=file,n_groups=len(g))
   for metric in ["mse","mae","mape"]:
    v=g[metric+"_improvement_pct"].to_numpy();record[metric+"_improved_groups"]=int((v>1e-9).sum());record[metric+"_worse_groups"]=int((v< -1e-9).sum());record[metric+"_unchanged_groups"]=int((np.abs(v)<=1e-9).sum())
   stability.append(record)
pd.DataFrame(stability).to_csv(R/"stability_summary.csv",index=False);print("Separated main, ablation and stability tables written")

import json
asset_rows=[]
for manifest in (ROOT/"data").glob("*/manifest.json"):
    ds=manifest.parent.name;idx=pd.read_csv(manifest.parent/"index.csv");idx=idx[idx.test]
    for asset,g in idx.groupby("asset"):
        count=int((g.events>0).sum());asset_rows.append(dict(dataset=ds,asset=asset,test_origins=len(g),event_origins=count,event_metrics_available=bool(count),comment="No eligible test events; all-origin metrics still reported" if not count else ("Descriptive small event sample" if count<20 else "")))
pd.DataFrame(asset_rows).to_csv(R/"asset_coverage.csv",index=False)

# Explain when the correction activates; include all test event origins, including zero gains.
mechanism=[]
for marker in sorted((ROOT/"evaluation").glob("*/*/COMPLETE")):
    folder=marker.parent;ds=folder.parent.name;base,seed=folder.name.rsplit("_s",1)
    meta=json.loads((folder/"verification.json").read_text())
    if base in ["TimesNet","Time-LLM"] and not meta["base_provenance"].get("causal_inference"):continue
    index=pd.read_csv(folder/"test_index.csv");tr=pd.read_csv(folder/"update_trace.csv")
    test=index[["asset","timestamp"]].rename(columns={"timestamp":"origin"})
    tr=tr.merge(test,on=["asset","origin"],how="inner",validate="many_to_one")
    for variant,v in tr.groupby("variant"):
        for asset in list(v.asset.unique())+["Pooled"]:
            q=v if asset=="Pooled" else v[v.asset==asset]
            active=(q.gain.abs()>1e-14)&(q.effect_l2>1e-14)
            mechanism.append(dict(dataset=ds,backbone=base,seed=int(seed),variant=variant,asset=asset,event_origins=len(q),active_origins=int(active.sum()),active_fraction=float(active.mean()),median_memory_n=float(q.memory_n.median()),median_calibration_n=float(q.calibration_n.median()),median_gain=float(q.gain.median()),maximum_gain=float(q.gain.max())))
pd.DataFrame(mechanism).to_csv(R/"mechanism_diagnostics.csv",index=False)

# Paired seed means, without treating random seeds as independent event samples.
horizon_data=pd.read_csv(R/"horizon_stability.csv");seed_mean_horizons=[]
for base in baselines:
    full="ARM90" if base=="Persistence" else base+"+ARM90"
    pair=horizon_data[horizon_data.method==full].merge(horizon_data[horizon_data.method==base],on=["dataset","asset","subset","seed","horizon"],suffixes=("_full","_base"),validate="one_to_one")
    if pair.empty:continue
    keys=["dataset","asset","subset","horizon"]
    cols=[m+suffix for m in ["mse","mae","mape"] for suffix in ["_full","_base"]]
    avg=pair.groupby(keys)[cols].mean().reset_index();counts=pair.groupby(keys).seed.nunique().rename("n_seeds").reset_index();avg=avg.merge(counts,on=keys);avg["reference"]=base
    for m in ["mse","mae","mape"]:avg[m+"_improvement_pct"]=100*(1-avg[m+"_full"]/avg[m+"_base"].clip(lower=1e-16))
    seed_mean_horizons.append(avg)
sh=pd.concat(seed_mean_horizons,ignore_index=True);sh.to_csv(R/"seed_mean_horizon_stability.csv",index=False);summary=[]
for vals,q in sh.groupby(["dataset","asset","subset","reference","n_seeds"]):
    rec=dict(zip(["dataset","asset","subset","reference","n_seeds"],vals));rec["horizons"]=len(q)
    for m in ["mse","mae","mape"]:
        v=q[m+"_improvement_pct"];rec[m+"_improved_horizons"]=int((v>1e-9).sum());rec[m+"_worse_horizons"]=int((v< -1e-9).sum());rec[m+"_unchanged_horizons"]=int((abs(v)<=1e-9).sum())
    summary.append(rec)
pd.DataFrame(summary).to_csv(R/"seed_mean_horizon_summary.csv",index=False)
