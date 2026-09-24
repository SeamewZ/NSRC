"""Read measured reports and draw temporal/horizon/ablation evidence."""
from pathlib import Path
import numpy as np,pandas as pd
from reproduction.full_report import R,plt
f=pd.read_csv(R/"metrics.csv");t=pd.read_csv(R/"temporal_stability.csv");h=pd.read_csv(R/"horizon_stability.csv");ci=pd.read_csv(R/"significance.csv")
for ds in f.dataset.unique():
 out=R/"figures"/ds;out.mkdir(parents=True,exist_ok=True)
 for base,seed in [("Persistence",0),("DLinear",2026)]:
  full="ARM90" if base=="Persistence" else base+"+ARM90";colors={base:"#CC6677",full:"#228833"}
  fig,axs=plt.subplots(2,2,figsize=(7,5.5),layout="constrained")
  for col,m in enumerate(["mse","mae"]):
   for method in [base,full]:
    hh=h[(h.dataset==ds)&(h.method==method)&(h.seed==seed)&(h.subset=="event")&(h.asset=="Pooled")].sort_values("horizon")
    tt=t[(t.dataset==ds)&(t.method==method)&(t.seed==seed)&(t.subset=="event")&(t.asset=="Pooled")&(t.scale=="month")].sort_values("period")
    axs[0,col].plot(hh.horizon,hh[m],label=method,color=colors[method]);axs[1,col].plot(tt.period,tt[m],"o-",label=method,color=colors[method]);axs[0,col].set(xlabel="Forecast step",ylabel=m.upper(),title="By horizon");axs[1,col].set(xlabel="Month",ylabel=m.upper(),title="By calendar period")
   axs[0,col].grid(alpha=.15);axs[1,col].grid(alpha=.15);axs[1,col].tick_params(axis="x",rotation=30)
  axs[0,0].legend(frameon=False);fig.savefig(out/f"stability_{base}.pdf");fig.savefig(out/f"stability_{base}.png",dpi=300);plt.close(fig)
  variants=[full]+([n for n in f.method.unique() if n.startswith("w_o_")] if base=="Persistence" else [n for n in f.method.unique() if n.startswith(base+"+w_o_")])
  a=f[(f.dataset==ds)&f.method.isin(variants)&(f.seed==seed)&(f.asset=="Pooled")&(f.subset=="event")&(f.horizon=="avg24")].copy()
  if len(a):
   fig,axs=plt.subplots(1,2,figsize=(7,3.5),layout="constrained");names=[n.replace(base+"+","").replace("w_o_","w/o ").replace("_"," ") for n in a.method]
   for ax,m in zip(axs,["mse","mae"]):
    ref=float(a.loc[a.method==full,m].iloc[0]);v=100*(a[m].to_numpy()/ref-1);ax.barh(names,v,color=["#228833" if n==full else "#4477AA" for n in a.method]);ax.axvline(0,color="black",lw=.5);ax.set_xlabel(m.upper()+" change vs full (%)");ax.invert_yaxis()
   fig.savefig(out/f"ablation_{base}.pdf");fig.savefig(out/f"ablation_{base}.png",dpi=300);plt.close(fig)
 # Paired CIs for the actual same-backbone comparisons, rather than selected best seeds.
 c=ci[(ci.dataset==ds)&(ci.family=="same_backbone")&(ci.reference_seed==2026)&(ci.asset=="Pooled")&(ci.subset=="event")&(ci.block_days==7)]
 if len(c):
  fig,axs=plt.subplots(1,3,figsize=(9,3.7),layout="constrained")
  for ax,m in zip(axs,["mse","mae","mape"]):
   z=c[c.metric==m];positions=np.arange(len(z));ax.hlines(positions,z.ci_low,z.ci_high,color="#4477AA");ax.scatter(z.delta,positions,color="#4477AA",s=18);ax.set_yticks(positions,z.reference);ax.axvline(0,color="black",lw=.7);ax.set_title(m.upper()+": baseline − corrected");ax.invert_yaxis()
  fig.savefig(out/"paired_confidence_intervals.pdf");fig.savefig(out/"paired_confidence_intervals.png",dpi=300);plt.close(fig)
print("Stability and CI figures written",flush=True)
