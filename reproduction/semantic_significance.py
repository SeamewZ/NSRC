"""Paired block bootstrap for semantic NSRC90 versus agnostic ARM90."""
from pathlib import Path
import numpy as np,pandas as pd
from reproduction.full_report import ROOT,losses,weights
R=ROOT/"reports";out=[]
for ds in ["celebrity","news","macro_crypto","macro_equities"]:
  for base,seed in [("DLinear",2026),("DLinear",2027),("DLinear",2028),("Persistence",0)]:
    folder=ROOT/"evaluation"/ds/f"{base}_s{seed}"
    if not (folder/"COMPLETE").exists():continue
    a=np.load(folder/"predictions.npz");idx=pd.read_csv(folder/"test_index.csv");keep=idx.events.to_numpy()>0
    if "NSRC90" not in a or "ARM90" not in a:continue
    dif=losses(a["ARM90"],a["y"]).mean(1)-losses(a["NSRC90"],a["y"]).mean(1)
    dates=pd.to_datetime(idx.timestamp,utc=True).dt.floor("D");day=(dates-dates.min()).dt.days.to_numpy();ndays=int(day.max())+1
    for subset,mask in [("event",keep),("all",np.ones(len(idx),bool))]:
      if not mask.any():continue
      counts=np.bincount(day[mask],minlength=ndays);totals=np.stack([np.bincount(day[mask],weights=dif[mask,j],minlength=ndays) for j in range(3)],axis=1);obs=dif[mask].mean(0)
      for block in [3,7,14]:
        w=weights(ndays,block);den=w@counts;valid=den>0;boot=(w[valid]@totals)/den[valid,None];ci=np.quantile(boot,[.025,.975],axis=0);p=(1+(np.abs(boot-obs)>=np.abs(obs)).sum(0))/(len(boot)+1)
        for j,m in enumerate(["mse","mae","mape"]):out.append(dict(dataset=ds,backbone=base,seed=seed,subset=subset,block_days=block,metric=m,n=int(mask.sum()),delta=float(obs[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),p_two_sided=float(p[j]),valid_resamples=int(len(boot))))
pd.DataFrame(out).to_csv(R/"semantic_significance.csv",index=False)
print("wrote",len(out),"semantic bootstrap rows")
