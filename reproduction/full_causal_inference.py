"""Causal, batch-composition-independent inference for official TimesNet and Time-LLM."""
import argparse,hashlib,json,os,shutil,sys,time,types
from pathlib import Path
import numpy as np,torch
ROOT=Path("runs/full_evaluation_20260921")
def timesnet_per_example(model):
 def forward(self,x):
  B,T,N=x.shape;xf=torch.fft.rfft(x,dim=1);amp=xf.abs().mean(-1);amp[:,0]=0;indices=torch.topk(amp,self.k,dim=1).indices;periods=T//indices
  scores=xf.abs().mean(-1).gather(1,indices);weights=torch.softmax(scores,dim=1);result=torch.zeros_like(x)
  for period in torch.unique(periods).tolist():
   active=(periods==period).any(1);ids=torch.nonzero(active).flatten();xx=x[ids];length=((T+period-1)//period)*period
   if length>T:xx=torch.cat([xx,torch.zeros(len(ids),length-T,N,device=x.device,dtype=x.dtype)],dim=1)
   panel=xx.reshape(len(ids),length//period,period,N).permute(0,3,1,2).contiguous();out=self.conv(panel);out=out.permute(0,2,3,1).reshape(len(ids),length,N)[:,:T]
   weight=(weights[ids]*(periods[ids]==period)).sum(1);result[ids]+=out*weight[:,None,None]
  return result+x
 originals=[]
 for block in model.model:
  originals.append(block.forward);block.forward=types.MethodType(forward,block)
 return originals
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--model",choices=["TimesNet","Time-LLM"],required=True);ap.add_argument("--seed",type=int,default=2026);a=ap.parse_args();ds=a.dataset;name=a.model;folder=ROOT/"baselines"/ds/f"{name}_s{a.seed}"
 if (folder/"CAUSAL_INFERENCE_COMPLETE").exists():return
 assert (folder/"COMPLETE").exists();meta=json.loads((folder/"provenance.json").read_text());checkpoint=folder
 if ds=="news":checkpoint=ROOT/"baselines/celebrity"/f"{name}_s{a.seed}"
 state=torch.load(checkpoint/"model.pt",map_location="cpu",weights_only=False);torch.set_num_threads(4)
 if name=="TimesNet":
  from reproduction.train import build
  model,predict,_=build(name,Path("vendor").resolve(),torch.device("cuda"));x=np.load(ROOT/"data"/ds/"windows.npz")["x"];model.load_state_dict(state["state_dict"]);model.eval();examples=np.linspace(0,len(x)-1,12,dtype=int);original=[]
  with torch.no_grad():
   for i in examples:original.append(predict(torch.from_numpy(x[i:i+1]).cuda()).cpu().numpy()[0])
  old_forwards=timesnet_per_example(model)
  with torch.no_grad():
   paired=predict(torch.from_numpy(x[examples]).cuda()).cpu().numpy();grouped_ok=np.allclose(paired,np.asarray(original),rtol=1e-4,atol=1e-5);pred=[]
   if grouped_ok:
    for lo in range(0,len(x),64):pred.append(predict(torch.from_numpy(x[lo:lo+64]).cuda()).cpu().numpy())
    out=np.concatenate(pred);grouped_ok=np.allclose(out[examples],np.asarray(original),rtol=1e-4,atol=1e-5)
   if not grouped_ok:
    for block,old in zip(model.model,old_forwards):block.forward=old
    out=np.empty((len(x),24),dtype=np.float32)
    for i in range(len(x)):
     out[i]=predict(torch.from_numpy(x[i:i+1]).cuda()).cpu().numpy()[0]
     if i%2000==0:print("TimesNet singleton fallback",ds,i,flush=True)
  np.testing.assert_allclose(out[examples],np.asarray(original),rtol=1e-4,atol=1e-5);maxerr=float(np.max(np.abs(out[examples]-original)));description="Per-example FFT periods, checked against upstream singleton inference" if grouped_ok else "Strict upstream singleton inference after grouped numerical-check fallback"
 else:
  from types import SimpleNamespace
  sys.path.insert(0,str(Path("vendor/Time-LLM").resolve()));from models.TimeLLM import Model
  cfg=SimpleNamespace(**state["provenance"]["config"]);model=Model(cfg).cuda();model.load_state_dict(state["state_dict"]);model.eval();x=np.load(ROOT/"data"/ds/"windows.npz")["x"][:,:,3:4].copy();predict=lambda xb:model(xb,None,None,None)[:,:,0]
  # Upstream singleton reference, before caching the input-independent vocabulary projection.
  examples=np.linspace(0,len(x)-1,12,dtype=int);single=[]
  with torch.no_grad():
   for i in examples:
    with torch.autocast("cuda",dtype=torch.bfloat16):p=predict(torch.from_numpy(x[i:i+1]).cuda())
    single.append(p.float().cpu().numpy()[0])
   with torch.autocast("cuda",dtype=torch.bfloat16):constant=model.mapping_layer(model.word_embeddings.permute(1,0)).detach()
  class FrozenProjection(torch.nn.Module):
   def __init__(self,value):super().__init__();self.register_buffer("value",value)
   def forward(self,unused):return self.value
  uncached_mapping=model.mapping_layer;model.mapping_layer=FrozenProjection(constant);out=np.empty((len(x),24),dtype=np.float32)
  with torch.no_grad():
   for i in range(len(x)):
    with torch.autocast("cuda",dtype=torch.bfloat16):p=predict(torch.from_numpy(x[i:i+1]).cuda())
    out[i]=p.float().cpu().numpy()[0]
    if i%2000==0:print("singleton inference",ds,i,"/",len(x),flush=True)
  if not np.allclose(out[examples],np.asarray(single),rtol=1e-6,atol=1e-7):
   model.mapping_layer=uncached_mapping
   with torch.no_grad():
    for i in range(len(x)):
     with torch.autocast("cuda",dtype=torch.bfloat16):p=predict(torch.from_numpy(x[i:i+1]).cuda())
     out[i]=p.float().cpu().numpy()[0]
     if i%2000==0:print("Uncached singleton fallback",ds,i,flush=True)
  np.testing.assert_allclose(out[examples],np.asarray(single),rtol=1e-6,atol=1e-7);maxerr=float(np.max(np.abs(out[examples]-single)));description="Strict singleton inference; only the input-independent vocabulary projection is cached. Verified against unmodified upstream singleton predictions."

 assert out.shape==(len(x),24) and np.isfinite(out).all();archive=ROOT/"diagnostics"/"batch_dependent_inference"/ds/folder.name;archive.mkdir(parents=True,exist_ok=True)
 for file in ["predictions.npy","provenance.json"]:
  if not (archive/file).exists():shutil.copy2(folder/file,archive/file)
 np.save(folder/"predictions.causal.npy",out);os.replace(folder/"predictions.causal.npy",folder/"predictions.npy")
 meta.update(causal_inference=description,singleton_max_abs_error=maxerr,index_sha256=hashlib.sha256((ROOT/"data"/ds/"index.csv").read_bytes()).hexdigest());(folder/"provenance.json").write_text(json.dumps(meta,indent=2))
 evaluation=ROOT/"evaluation"/ds/folder.name
 if evaluation.exists():
  while not ((evaluation/"COMPLETE").exists() or (evaluation/"FAILED").exists()):time.sleep(5)
  dest=archive/"evaluation_before_causal_inference"
  if not dest.exists():shutil.move(str(evaluation),str(dest))
 (folder/"CAUSAL_INFERENCE_COMPLETE").write_text(json.dumps(dict(checked_singletons=len(examples),max_abs_error=maxerr,description=description),indent=2));print("CAUSAL INFERENCE COMPLETE",ds,name,a.seed,maxerr,flush=True)
if __name__=="__main__":main()
