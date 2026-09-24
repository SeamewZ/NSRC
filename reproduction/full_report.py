"""Measured metrics, calendar-block bootstrap, stability and complete price-case gallery."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path("runs/full_evaluation_20260921");R=ROOT/"reports"
plt.rcParams.update({"font.size":9,"axes.titlesize":9,"axes.labelsize":9,"legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,"pdf.fonttype":42,"svg.fonttype":"none"})
METRICS=["mse","mae","mape"]
def losses(y,p):
    d=p-y;return np.stack([d*d,np.abs(d),100*np.abs(d)/np.maximum(np.abs(y),1e-8)],axis=-1)
def load(ds):
    packs={};idx=None;y=None
    for flag in sorted((ROOT/"evaluation"/ds).glob("*/COMPLETE")):
        folder=flag.parent;base,seed=folder.name.rsplit("_s",1)
        if base in ["TimesNet","Time-LLM"]:
            provenance=json.loads((folder/"verification.json").read_text())["base_provenance"]
            if not provenance.get("causal_inference"):continue
        a=np.load(folder/"predictions.npz");current=pd.read_csv(folder/"test_index.csv")
        if idx is None:idx=current;y=a["y"]
        else:assert np.array_equal(y,a["y"]) and idx.equals(current)
        for variant in a.files:
            if variant=="y":continue
            name=("Persistence" if variant=="no_correction" else variant) if base=="Persistence" else (base if variant=="no_correction" else base+"+"+variant)
            packs[(name,int(seed))]=a[variant]
    return idx,y,packs

def metrics(ds,idx,y,packs):
    rows=[];periods=[];horizons=[];vol=[];wins=[]
    base_idx=pd.read_csv(ROOT/"data"/ds/"index.csv");sigma=np.load(ROOT/"data"/ds/"features.npz")["sigma"];test_sigma=sigma[base_idx.test.to_numpy()];dev=~base_idx.test.to_numpy()&(pd.to_datetime(base_idx.target_end,utc=True)<pd.to_datetime(idx.timestamp,utc=True).min()).to_numpy()
    cut=np.quantile(sigma[dev],[1/3,2/3]);bucket=np.digitize(test_sigma,cut);week=pd.to_datetime(idx.timestamp,utc=True).dt.strftime("%G-W%V").to_numpy();assets=list(idx.asset.unique())+["Pooled"]
    times=pd.to_datetime(idx.timestamp,utc=True);midpoint=times.min()+(times.max()-times.min())/2;segments=np.where(times<=midpoint,"early","late")
    for (name,seed),p in packs.items():
        l=losses(y,p)
        for asset in assets:
            am=np.ones(len(idx),bool) if asset=="Pooled" else idx.asset.to_numpy()==asset
            for subset,keep in [("all",am),("event",am&(idx.events.to_numpy()>0))]:
                if not keep.any():continue
                n=int(keep.sum());base=dict(dataset=ds,method=name,seed=seed,asset=asset,subset=subset,n=n)
                for h,lo,hi in [("avg24",0,24),("1-6",0,6),("7-12",6,12),("13-24",12,24)]:rows.append(dict(base,horizon=h,**dict(zip(METRICS,l[keep,lo:hi].mean((0,1))))))
                for h in range(24):horizons.append(dict(base,horizon=h+1,**dict(zip(METRICS,l[keep,h].mean(0)))))
                for scale,values in [("month",idx.month.to_numpy()),("week",week),("segment",segments)]:
                    for period in sorted(set(values[keep])):
                        k=keep&(values==period);periods.append(dict(base,scale=scale,period=period,n=int(k.sum()),**dict(zip(METRICS,l[k].mean((0,1))))))
                for b,label in enumerate(["low","medium","high"]):
                    k=keep&(bucket==b)
                    if k.any():vol.append(dict(base,volatility=label,n=int(k.sum()),lower_cut=float(cut[0]),upper_cut=float(cut[1]),**dict(zip(METRICS,l[k].mean((0,1))))))
                references=[("Persistence",0)] if ("Persistence",0) in packs else []
                if "+ARM90" in name and (name.replace("+ARM90",""),seed) in packs:references.append((name.replace("+ARM90",""),seed))
                for ref in references:
                    delta=losses(y[keep],packs[ref][keep]).mean(1)-l[keep].mean(1)
                    for j,m in enumerate(METRICS):wins.append(dict(base,reference=ref[0],reference_seed=ref[1],metric=m,wins=int((delta[:,j]>1e-12).sum()),ties=int((np.abs(delta[:,j])<=1e-12).sum()),losses=int((delta[:,j]<-1e-12).sum())))
    return rows,periods,horizons,vol,wins

BOOT={}
def weights(n,block):
    key=n,block
    if key not in BOOT:
        rng=np.random.default_rng(20260921+block+n);starts=rng.integers(0,n,size=(49999,int(np.ceil(n/block))));indices=((starts[:,:,None]+np.arange(block))%n).reshape(49999,-1)[:,:n];w=np.zeros((49999,n),dtype=np.float64)
        np.add.at(w,(np.arange(49999)[:,None],indices),1);BOOT[key]=w
    return BOOT[key]
def bootstrap(ds,idx,y,packs):
    rows=[];dates=pd.to_datetime(idx.timestamp,utc=True).dt.floor("D");day=((dates-dates.min()).dt.days).to_numpy();ndays=int(day.max())+1;assets=list(idx.asset.unique())+["Pooled"]
    pairs=[]
    if ("ARM90",0) not in packs:return rows
    for key in packs:
        name,seed=key
        if name in ["ARM90"]:continue
        if seed in [0,2026]:
            if name.startswith("w_o_") or name.startswith("window") or name in ["topic90","context90"]:pairs.append((key,("ARM90",0),"ablation_vs_full"))
            elif "+" not in name:pairs.append((("ARM90",0),key,"primary_vs_baseline"))
            elif any(tag in name for tag in ["+w_o_","+window","+topic90","+context90"]):pairs.append((key,(name.split("+")[0]+"+ARM90",seed),"backbone_ablation_vs_full"))
        if "+ARM90" in name:pairs.append((key,(name.replace("+ARM90",""),seed),"same_backbone"))
    for candidate,reference,family in pairs:
        if reference not in packs:continue
        dif=losses(y,packs[reference]).mean(1)-losses(y,packs[candidate]).mean(1)
        for asset in assets:
            am=np.ones(len(idx),bool) if asset=="Pooled" else idx.asset.to_numpy()==asset
            for subset,keep in [("event",am&(idx.events.to_numpy()>0)),("all",am)]:
                if not keep.any():continue
                count=np.bincount(day[keep],minlength=ndays);total=np.stack([np.bincount(day[keep],weights=dif[keep,j],minlength=ndays) for j in range(3)],axis=1)
                n=int(keep.sum());active=int((count>0).sum());obs=dif[keep].mean(0)
                for block in [3,7,14]:
                    w=weights(ndays,block);den=w@count;valid=den>0;boot=(w[valid]@total)/den[valid,None];ci=np.quantile(boot,[.025,.975],axis=0);p=(1+(np.abs(boot-obs)>=np.abs(obs)).sum(0))/(len(boot)+1)
                    for j,m in enumerate(METRICS):rows.append(dict(dataset=ds,asset=asset,subset=subset,candidate=candidate[0],candidate_seed=candidate[1],reference=reference[0],reference_seed=reference[1],family=family,metric=m,n=n,event_days=active,block_days=block,delta=float(obs[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),p_two_sided=float(p[j]),valid_resamples=len(boot),inferential=bool(n>=20 and active>=8),status="exploratory, conditional on fitted online trajectory"))
    return rows

def draw_case(ax,idx,y,base,p,i,units):
    scale=float(idx.iloc[i].origin_close);h=np.arange(1,25);yb=y[i]*scale;bb=base[i]*scale;pp=p[i]*scale
    ax.plot(h,yb,"s-",color="#243B53",ms=2,lw=1.15,label="Observed")
    ax.plot(h,bb,":",color="#CC6677",lw=1.4,label="Baseline")
    ax.plot(h,pp,"o-",color="#228833",ms=2,lw=1.1,label="Corrected")
    ax.fill_between(h,bb,pp,color="#228833",alpha=.12)
    before=np.mean(np.abs(base[i]-y[i])/np.maximum(np.abs(y[i]),1e-8))*100;after=np.mean(np.abs(p[i]-y[i])/np.maximum(np.abs(y[i]),1e-8))*100
    date=str(idx.iloc[i].timestamp)[:16];ax.set_title(f"{idx.iloc[i].asset} | {date} UTC\nMAPE: {before:.2f}% → {after:.2f}%",loc="left")
    ax.set(xlabel=units,ylabel="Index level" if idx.iloc[i].asset in ["SPX","NDX","INDU"] else "Price (USD)",xticks=[1,6,12,18,24]);ax.grid(alpha=.15);ax.ticklabel_format(axis="y",style="plain",useOffset=False)

def plots(ds,idx,y,packs):
    if ("ARM90",0) not in packs:return
    out=R/"figures"/ds;out.mkdir(parents=True,exist_ok=True);csvdir=R/"curves"/ds;csvdir.mkdir(parents=True,exist_ok=True)
    base=packs[("Persistence",0)];p=packs[("ARM90",0)];ev=np.flatnonzero(idx.events.to_numpy()>0);manifest=json.loads((ROOT/"data"/ds/"manifest.json").read_text());units="Forecast horizon (hours)" if ds!="macro_equities" else "Forecast horizon (trading steps)"
    improvement=np.mean(np.abs(base-y)-np.abs(p-y),axis=1);ordered=ev[np.argsort(improvement[ev],kind="stable")];selected=ordered[np.rint(np.asarray([.25,.5,.75])*(len(ordered)-1)).astype(int)]
    # Keep quartiles fixed; all unsuccessful cases remain in the complete gallery.
    fig,axs=plt.subplots(3,1,figsize=(3.5,7.4),layout="constrained")
    for ax,i in zip(axs,selected):draw_case(ax,idx,y,base,p,i,units)
    fig.legend(*axs[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False,handlelength=1.3,columnspacing=.8);fig.savefig(out/"fig2_ARM90.pdf");fig.savefig(out/"fig2_ARM90.png",dpi=300);fig.savefig(out/"fig2_ARM90.svg");plt.close(fig)
    selection=[]
    for i in selected:selection.append(dict(dataset=ds,row=int(i),asset=idx.iloc[i].asset,origin=idx.iloc[i].timestamp,selection="25/50/75 percentiles of MAE improvement; outcome-stratified illustration",mae_improvement=float(improvement[i])))
    earliest=idx.iloc[ev].reset_index().groupby(["asset","month"],sort=True).head(1)["index"].to_numpy()
    with PdfPages(out/"all_event_curves.pdf") as pdf:
        for start in range(0,len(ev),6):
            fig,axs=plt.subplots(3,2,figsize=(10,10),layout="constrained")
            for ax,i in zip(axs.flat,ev[start:start+6]):draw_case(ax,idx,y,base,p,i,units)
            for ax in list(axs.flat)[len(ev[start:start+6]):]:ax.set_visible(False)
            fig.legend(*axs.flat[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False);pdf.savefig(fig);plt.close(fig)
    with PdfPages(out/"earliest_asset_month_cases.pdf") as pdf:
        for i in earliest:
            fig,ax=plt.subplots(figsize=(5,3.3),layout="constrained");draw_case(ax,idx,y,base,p,i,units);ax.legend(frameon=False);pdf.savefig(fig);plt.close(fig)
    # Complete long-form curves for every method, preserving per-seed predictions.
    scale=idx.origin_close.to_numpy()[ev];curves=[]
    for (method,seed),pred in packs.items():
        if method!="ARM90" and "+ARM90" not in method:continue
        ref="Persistence" if method=="ARM90" else method.replace("+ARM90","");b=packs[(ref,seed)]
        frame=pd.DataFrame(dict(row=np.repeat(ev,24),asset=np.repeat(idx.iloc[ev].asset.to_numpy(),24),origin=np.repeat(idx.iloc[ev].timestamp.to_numpy(),24),horizon=np.tile(np.arange(1,25),len(ev)),observed=(y[ev]*scale[:,None]).ravel(),baseline=(b[ev]*scale[:,None]).ravel(),corrected=(pred[ev]*scale[:,None]).ravel()))
        frame.to_csv(csvdir/f"{method}_s{seed}.csv.gz",index=False)
    pd.DataFrame(selection).to_csv(out/"case_selection.csv",index=False)
    for backbone in ["DLinear","GPT4TS","PatchTST"]:
        key=(backbone+"+ARM90",2026)
        if key not in packs:continue
        bp=packs[(backbone,2026)];cp=packs[key];gain=np.mean(np.abs(bp-y)-np.abs(cp-y),axis=1);order=ev[np.argsort(gain[ev],kind="stable")];chosen=order[np.rint(np.asarray([.25,.5,.75])*(len(order)-1)).astype(int)]
        fig,axs=plt.subplots(3,1,figsize=(3.5,7.4),layout="constrained")
        for ax,i in zip(axs,chosen):draw_case(ax,idx,y,bp,cp,i,units)
        fig.legend(*axs[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False,handlelength=1.3,columnspacing=.8)
        for ext in ["pdf","svg","png"]:fig.savefig(out/f"fig2_{backbone}_ARM90.{ext}",dpi=300)
        plt.close(fig);pd.DataFrame([dict(row=int(i),asset=idx.iloc[i].asset,origin=idx.iloc[i].timestamp,selection="outcome-stratified quartile illustration",mae_improvement=float(gain[i])) for i in chosen]).to_csv(out/f"case_selection_{backbone}.csv",index=False)
        if backbone=="DLinear":
            with PdfPages(out/"all_event_curves_DLinear.pdf") as pdf:
                for start in range(0,len(ev),6):
                    fig,axs=plt.subplots(3,2,figsize=(10,10),layout="constrained")
                    for ax,i in zip(axs.flat,ev[start:start+6]):draw_case(ax,idx,y,bp,cp,i,units)
                    for ax in list(axs.flat)[len(ev[start:start+6]):]:ax.set_visible(False)
                    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False);pdf.savefig(fig);plt.close(fig)
    (out/"MANIFEST.json").write_text(json.dumps(dict(primary_backbone="Persistence",default_gallery="all_event_curves.pdf",dlinear_gallery="all_event_curves_DLinear.pdf",neural_seed=2026,selection="quartile illustrative examples; complete galleries include every event origin"),indent=2))


def main(make_plots=True):
    R.mkdir(parents=True,exist_ok=True);allrows=[[] for _ in range(5)];cis=[];coverage=[];loaded={}
    for ds in ["celebrity","news","macro_crypto","macro_equities"]:
        idx,y,packs=load(ds)
        if idx is None:continue
        loaded[ds]=(idx,y,packs);parts=metrics(ds,idx,y,packs)
        for dest,part in zip(allrows,parts):dest.extend(part)
        cis.extend(bootstrap(ds,idx,y,packs))
        for method in ["Persistence","ARM90","DLinear","GPT4TS","PatchTST","iTransformer","TimesNet","Time-LLM","TEST","CAMEF"]:
            seeds=sorted(seed for name,seed in packs if name==method);coverage.append(dict(dataset=ds,method=method,seeds=",".join(map(str,seeds)),n_seeds=len(seeds),status="measured" if seeds else ("blocked_core_implementation" if method=="TEST" else "pending_or_failed")))
        if make_plots:plots(ds,idx,y,packs)
        print("reported",ds,len(packs),flush=True)
    for name,rows in zip(["metrics","temporal_stability","horizon_stability","volatility_stability","win_tie_loss"],allrows):pd.DataFrame(rows).to_csv(R/f"{name}.csv",index=False)
    frame=pd.DataFrame(allrows[0]);keys=["dataset","method","asset","subset","horizon"]
    agg=frame.groupby(keys,sort=False).agg(n_seeds=("seed","nunique"),n=("n","first"),mse_mean=("mse","mean"),mse_std=("mse","std"),mae_mean=("mae","mean"),mae_std=("mae","std"),mape_mean=("mape","mean"),mape_std=("mape","std")).reset_index();agg.to_csv(R/"seed_summary.csv",index=False)
    ci=pd.DataFrame(cis);ci["p_holm"]=np.nan
    # One prespecified family over all eligible primary 7-day event tests across data/assets/metrics/references.
    select=ci.index[(ci.family=="primary_vs_baseline")&(ci.block_days==7)&(ci.subset=="event")&ci.inferential].to_numpy();order=select[np.argsort(ci.loc[select,"p_two_sided"].to_numpy())];adj=np.maximum.accumulate(np.minimum(1,ci.loc[order,"p_two_sided"].to_numpy()*(len(order)-np.arange(len(order)))));ci.loc[order,"p_holm"]=adj
    ci["holm_family"]=ci.family
    for family in ["same_backbone","ablation_vs_full","backbone_ablation_vs_full"]:
        ids=ci.index[(ci.family==family)&(ci.block_days==7)&(ci.subset=="event")&ci.inferential].to_numpy();order=ids[np.argsort(ci.loc[ids,"p_two_sided"].to_numpy())]
        if len(order):ci.loc[order,"p_holm"]=np.maximum.accumulate(np.minimum(1,ci.loc[order,"p_two_sided"].to_numpy()*(len(order)-np.arange(len(order)))))
    ci.to_csv(R/"significance.csv",index=False)
    pd.DataFrame(coverage).to_csv(R/"coverage.csv",index=False)
    individual=frame[frame.asset!="Pooled"]
    individual.groupby(["dataset","method","seed","subset","horizon"]).agg(n_assets=("asset","nunique"),mse=("mse","mean"),mae=("mae","mean"),mape=("mape","mean")).reset_index().to_csv(R/"equal_asset_macro_average.csv",index=False)
    improvement=[]
    for ds,(ix,truth,pks) in loaded.items():
        for (name,seed),pred in pks.items():
            if name!="ARM90" and "+ARM90" not in name:continue
            ref="Persistence" if name=="ARM90" else name.replace("+ARM90","");current=frame[(frame.dataset==ds)&(frame.method==name)&(frame.seed==seed)&(frame.horizon=="avg24")];base=frame[(frame.dataset==ds)&(frame.method==ref)&(frame.seed==seed)&(frame.horizon=="avg24")];merged=current.merge(base,on=["dataset","asset","subset","seed","horizon"],suffixes=("_corrected","_base"))
            for _,row in merged.iterrows():
                result=dict(dataset=ds,method=name,reference=ref,seed=seed,asset=row.asset,subset=row.subset)
                for m in METRICS:result[m+"_improvement_pct"]=100*(1-row[m+"_corrected"]/row[m+"_base"])
                improvement.append(result)
    pd.DataFrame(improvement).to_csv(R/"correction_improvements.csv",index=False)
    main=agg[(agg.horizon=="avg24")&(agg.subset=="event")];main.to_csv(R/"main_event_table.csv",index=False)
    # Paper-friendly values are generated only from these measured arrays, never copied from manuscript tables.
    main[main.asset!="Pooled"].to_latex(R/"main_event_table.tex",index=False,float_format=lambda x:f"{x:.6f}")
    text=["# Full evaluation (exploratory)\n","These are measured, prospective-aligned forecasts. Reused evaluation periods are exploratory.\n","MSE and MAE use prices divided by origin close; MAPE is percent. Average is across all 24 future observations, not proof of per-period stability.\n","Primary ARM90 uses persistence plus event-triggered agnostic residual memory. Topic/context controls are reported separately.\n","Crypto horizons are hours; equity horizons are 24 trading observations. Event and all-origin tables are separate.\n","Block bootstrap CIs are conditional on the fitted online trajectory; the 7-day primary event family uses Holm adjustment. N<20 or <8 event days is descriptive.\n","Seed standard deviations are blank for a single seed, not zero. Full curve galleries include failures. Fig.2 selects improvement quartiles and is explicitly illustrative.\n","## Coverage\n",pd.DataFrame(coverage).to_csv(index=False),"\n## Pooled event results\n",main[main.asset=="Pooled"].to_csv(index=False)]
    (R/"README.md").write_text("\n".join(text));(R/"GENERATED").write_text(pd.Timestamp.now(tz="UTC").isoformat())
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--no-plots",action="store_true");args=p.parse_args();main(not args.no_plots)
