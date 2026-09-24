"""Verified reports for semantic NSRC90; preserves original ARM90 report tree."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from reproduction.semantic_online import ROOT,BASE
from reproduction.full_report import losses,weights,draw_case,plt
from matplotlib.backends.backend_pdf import PdfPages
R=ROOT/"reports";R.mkdir(exist_ok=True)
DATASETS=["celebrity","news","macro_crypto","macro_equities"]
NAMES={"celebrity":"Celebrity","news":"News","macro_crypto":"Macro crypto","macro_equities":"Macro equities"}
BASELINES=["Persistence","DLinear","GPT4TS","PatchTST","iTransformer","TimesNet","Time-LLM","CAMEF"]
LABELS={"NSRC90":"NSRC","ARM90":"w/o semantics","w_o_z":"w/o description","w_o_c":"w/o category","w_o_d":"w/o direction","w_o_fallback":"w/o fallback","w_o_calibration":"w/o calibration","w_o_online_update":"w/o updates","w_o_recency":"w/o recency","w_o_volatility":"w/o volatility","w_o_sharing":"w/o sharing","w_o_support_shrinkage":"w/o shrinkage","shuffled_memory":"shuffled keys"}
def load(ds):
 packs={};idx=None;y=None;audits=[]
 for flag in sorted((ROOT/"evaluation"/ds).glob("*/COMPLETE")):
  f=flag.parent;b,seed=f.name.rsplit("_s",1);a=np.load(f/"predictions.npz");ix=pd.read_csv(f/"test_index.csv")
  if idx is None:idx=ix;y=a["y"]
  else:assert idx.equals(ix);np.testing.assert_array_equal(y,a["y"])
  v=json.loads((f/"verification.json").read_text());assert all(all(c[k] for k in ["finite","no_event_identity","future_target_invariance"]) for c in v["checks"])
  for variant in a.files:
   if variant!="y":packs[(b,int(seed),variant)]=a[variant]
  audits.append(dict(dataset=ds,backbone=b,seed=int(seed),sha256=hashlib.sha256((f/"predictions.npz").read_bytes()).hexdigest(),variants=len(a.files)-1))
 return idx,y,packs,audits
def bootstrap(ds,idx,dif,base,reference,nseed,refseed,analysis):
 day=(pd.to_datetime(idx.timestamp,utc=True).dt.floor("D")-pd.to_datetime(idx.timestamp,utc=True).dt.floor("D").min()).dt.days.to_numpy();n=int(day.max())+1;keep=idx.events.to_numpy()>0
 count=np.bincount(day[keep],minlength=n);total=np.column_stack([np.bincount(day[keep],weights=dif[keep,j],minlength=n) for j in range(3)]);obs=dif[keep].mean(0);out=[]
 for block in [3,7,14]:
  w=weights(n,block);den=w@count;valid=den>0;boot=(w[valid]@total)/den[valid,None];ci=np.quantile(boot,[.025,.975],axis=0)
  p=(1+(np.abs(boot-obs)>=np.abs(obs)).sum(0))/(len(boot)+1)
  for j,m in enumerate(["mse","mae","mape"]):
   out.append(dict(dataset=ds,backbone=base,reference=reference,analysis=analysis,n_seeds=nseed,reference_seeds=refseed,metric=m,block_days=block,n=int(keep.sum()),active_days=int((count>0).sum()),delta=float(obs[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),p_two_sided=float(p[j]),inferential=bool(keep.sum()>=20 and (count>0).sum()>=8)))
 return out
def main():
 assert (ROOT/"MEASUREMENTS_COMPLETE").exists()
 metrics=[];horizons=[];periods=[];wins=[];cis=[];curvecount=0;audits=[];verified=0;loaded={}
 for ds in DATASETS:
  idx,y,packs,audit=load(ds);audits+=audit;loaded[ds]=(idx,y,packs)
  times=pd.to_datetime(idx.timestamp,utc=True);mid=times.min()+(times.max()-times.min())/2
  groups={"month":times.dt.strftime("%Y-%m").to_numpy(),"week":times.dt.strftime("%G-W%V").to_numpy(),"half":np.where(times<=mid,"early","late")}
  original_idx=pd.read_csv(BASE/"data"/ds/"index.csv");sig=np.load(BASE/"data"/ds/"features.npz")["sigma"]
  dev=(~original_idx.test.to_numpy())&(pd.to_datetime(original_idx.target_end,utc=True)<times.min()).to_numpy()
  cuts=np.quantile(sig[dev],[1/3,2/3]);groups["volatility"]=np.asarray(["low","medium","high"])[np.digitize(sig[original_idx.test.to_numpy()],cuts)]
  meanloss={}
  for (b,s,v),p in packs.items():
   ell=losses(y,p);ev=idx.events.to_numpy()>0
   if v=="NSRC90":
    for ref in ["no_correction","ARM90"]:
     delta=losses(y,packs[(b,s,ref)]).mean(1)-ell.mean(1)
     for j,m in enumerate(["mse","mae","mape"]):wins.append(dict(dataset=ds,backbone=b,seed=s,reference=ref,metric=m,n=int(ev.sum()),wins=int((delta[ev,j]>1e-12).sum()),ties=int((abs(delta[ev,j])<=1e-12).sum()),losses=int((delta[ev,j]<-1e-12).sum())))
   meanloss.setdefault((b,v),[]).append(ell.mean(1))
   for asset in ["Pooled",*idx.asset.unique()]:
    assetmask=np.ones(len(idx),bool) if asset=="Pooled" else idx.asset.to_numpy()==asset
    for subset,mask in [("event",assetmask&ev),("all",assetmask)]:
     if not mask.any():continue
     common=dict(dataset=ds,backbone=b,variant=v,seed=s,asset=asset,subset=subset,n=int(mask.sum()))
     score=ell[mask].mean((0,1))
     # Independent pooled mean verification, including the true-price MAPE denominator.
     if asset=="Pooled":
      error=p[mask]-y[mask];actual=[np.mean(error**2),np.mean(abs(error)),100*np.mean(abs(error)/np.maximum(abs(y[mask]),1e-8))]
      np.testing.assert_allclose(score,actual,rtol=1e-12,atol=1e-14);verified+=1
     metrics.append(dict(common,**dict(zip(["mse","mae","mape"],score))))
     if v in ["no_correction","ARM90","NSRC90"]:
      for h in range(24):horizons.append(dict(common,horizon=h+1,**dict(zip(["mse","mae","mape"],ell[mask,h].mean(0)))))
      for kind,values in groups.items():
       for period in sorted(set(values[mask])):
        k=mask&(values==period);periods.append(dict(common,group=kind,period=period,n=int(k.sum()),**dict(zip(["mse","mae","mape"],ell[k].mean((0,1))))))
  for b in BASELINES:
   if (b,"NSRC90") not in meanloss:continue
   full=np.mean(meanloss[(b,"NSRC90")],axis=0);nseed=len(meanloss[(b,"NSRC90")])
   for ref in ["ARM90","no_correction"]:
    dif=np.mean(meanloss[(b,ref)],axis=0)-full
    cis+=bootstrap(ds,idx,dif,b,ref,nseed,len(meanloss[(b,ref)]),"same_backbone")
   if b=="DLinear":
    for (bb,v),arr in meanloss.items():
     if bb=="DLinear" and v not in ["NSRC90","ARM90","no_correction"]:
      cis+=bootstrap(ds,idx,np.mean(arr,axis=0)-full,b,v,nseed,len(arr),"ablation")
  out=R/"figures"/ds;out.mkdir(parents=True,exist_ok=True);cd=R/"curves"/ds;cd.mkdir(parents=True,exist_ok=True);ev=np.flatnonzero(idx.events.to_numpy()>0)
  for (b,s,v),p in packs.items():
   if v!="NSRC90":continue
   scale=idx.origin_close.to_numpy()[ev];bb=packs[(b,s,"no_correction")]
   c=pd.DataFrame(dict(row=np.repeat(ev,24),asset=np.repeat(idx.iloc[ev].asset.to_numpy(),24),origin=np.repeat(idx.iloc[ev].timestamp.to_numpy(),24),horizon=np.tile(np.arange(1,25),len(ev)),observed=(y[ev]*scale[:,None]).ravel(),baseline=(bb[ev]*scale[:,None]).ravel(),corrected=(p[ev]*scale[:,None]).ravel()))
   path=cd/f"{b}_s{s}.csv.gz";c.to_csv(path,index=False);back=pd.read_csv(path)
   np.testing.assert_allclose(back.corrected.to_numpy(),(p[ev]*scale[:,None]).ravel(),rtol=1e-12,atol=1e-10);curvecount+=len(c)
  b=packs[("DLinear",2026,"no_correction")];p=packs[("DLinear",2026,"NSRC90")];units="Forecast horizon (trading steps)" if ds=="macro_equities" else "Forecast horizon (hours)"
  # First available test event for each asset, then earliest remaining event; no outcome-based ranking.
  selected=[]
  for asset in idx.asset.unique():
   ids=ev[idx.iloc[ev].asset.to_numpy()==asset]
   if len(ids):selected.append(int(ids[0]))
  for i in ev:
   if len(selected)>=3:break
   if i not in selected:selected.append(int(i))
  selected=selected[:3]
  fig,axes=plt.subplots(3,1,figsize=(3.5,6.8),layout="constrained")
  for ax,i in zip(axes,selected):draw_case(ax,idx,y,b,p,i,units)
  fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False,handlelength=1.)
  for ext in ["pdf","svg","png"]:fig.savefig(out/f"fig2_fixed.{ext}",dpi=300)
  plt.close(fig)
  pd.DataFrame([dict(row=i,asset=idx.iloc[i].asset,origin=idx.iloc[i].timestamp,rule="first test event per asset; asset coverage then chronological order, independent of errors") for i in selected]).to_csv(out/"case_selection.csv",index=False)
  with PdfPages(out/"all_event_curves.pdf") as pdf:
   for start in range(0,len(ev),6):
    fig,axes=plt.subplots(3,2,figsize=(10,10),layout="constrained")
    for ax,i in zip(axes.flat,ev[start:start+6]):draw_case(ax,idx,y,b,p,i,units)
    for ax in list(axes.flat)[len(ev[start:start+6]):]:ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False);pdf.savefig(fig);plt.close(fig)
  print("verified and reported",ds,flush=True)
 f=pd.DataFrame(metrics);f.to_csv(R/"metrics.csv",index=False)
 summary=f.groupby(["dataset","backbone","variant","asset","subset"]).agg(n_seeds=("seed","nunique"),n=("n","first"),mse_mean=("mse","mean"),mse_std=("mse","std"),mae_mean=("mae","mean"),mae_std=("mae","std"),mape_mean=("mape","mean"),mape_std=("mape","std")).reset_index()
 summary.to_csv(R/"seed_summary.csv",index=False)
 for name,rows in [("horizon_stability",horizons),("period_stability",periods),("win_tie_loss",wins)]:pd.DataFrame(rows).to_csv(R/f"{name}.csv",index=False)
 f[f.asset!="Pooled"].groupby(["dataset","backbone","variant","seed","subset"]).agg(n_assets=("asset","nunique"),mse=("mse","mean"),mae=("mae","mean"),mape=("mape","mean")).reset_index().to_csv(R/"equal_asset_average.csv",index=False)
 ci=pd.DataFrame(cis);ci["p_holm"]=np.nan;ci["family"]=""
 for ref in ["ARM90","no_correction"]:
  ids=ci.index[(ci.backbone=="DLinear")&(ci.reference==ref)&(ci.block_days==7)&ci.inferential].to_numpy()
  order=ids[np.argsort(ci.loc[ids,"p_two_sided"])];ci.loc[order,"p_holm"]=np.maximum.accumulate(np.minimum(1,ci.loc[order,"p_two_sided"].to_numpy()*(len(order)-np.arange(len(order)))));ci.loc[ids,"family"]="DLinear_four_cohort_three_metric_"+ref
 ids=ci.index[(ci.analysis=="ablation")&(ci.block_days==7)&ci.inferential].to_numpy();order=ids[np.argsort(ci.loc[ids,"p_two_sided"])]
 ci.loc[order,"p_holm"]=np.maximum.accumulate(np.minimum(1,ci.loc[order,"p_two_sided"].to_numpy()*(len(order)-np.arange(len(order)))));ci.loc[ids,"family"]="DLinear_all_ablations_four_cohorts"
 ci.to_csv(R/"significance.csv",index=False)
 assert ci.p_two_sided.between(1/50000,1).all() and (ci.ci_low<=ci.ci_high).all()
 verification=dict(evaluated_cells=len(audits),independently_checked_metric_rows=verified,verified_curve_rows=curvecount,all_variant_future_target_invariance=True,original_baseline_hashes_preserved=True,
  significance="Mean paired losses across common seeds first; calendar-day block bootstrap; MAPE denominator is observed normalized price",exploratory_reused_periods=True,
  missing_methods=["TEST"],cells=audits)
 (ROOT/"verification.json").write_text(json.dumps(verification,indent=2))
 # Correctly state which causal condition is demonstrated, without erasing source-time assumptions.
 text=["# NSRC90实验结果（探索性）","","四组分别统计；不把股指交易步和加密资产小时合为主结论。DLinear为既定重点骨干，所有原始基线均保留。","",
 "| 数据组 | 方法 | 种子数 | MSE | MAE | MAPE (%) |","|---|---|---:|---:|---:|---:|"]
 for ds in DATASETS:
  for variant in ["no_correction","ARM90","NSRC90"]:
   row=summary[(summary.dataset==ds)&(summary.backbone=="DLinear")&(summary.variant==variant)&(summary.asset=="Pooled")&(summary.subset=="event")].iloc[0]
   text.append(f"| {NAMES[ds]} | {variant} | {row.n_seeds} | {row.mse_mean:.8f} | {row.mae_mean:.8f} | {row.mape_mean:.5f} |")
 text+=["","语义收益以NSRC90对ARM90比较，不能用相对原骨干的总收益代替。主显著性对每个起点先平均种子损失，再按日块重采样；没有将三个种子视为三倍独立事件。无描述/无类别/无方向分别保留其余组件。无fallback若完全相同意味着此数据未实际触发该机制，不能声称它带来增益。","",
 "新稿需要报告负结果，不宣称所有方法、资产或种子均获益。TEST没有可用的完整官方训练链，未填数值。原始论文手填表和旧CI未用于本轮统计。","",
 "原新闻文本历史版本、宏观时间戳语义尚未独立核实；复用时段属于探索性评估。源码的时间约束检查只证明实现满足已设定的信息时间，并不能代替数据来源审计。"]
 (R/"RESULTS.md").write_text("\n".join(text)+"\n")
 (ROOT/"REPORT_COMPLETE").touch()
 print("FINAL",verified,curvecount,flush=True)
if __name__=="__main__":main()
