"""CAMEF dataset adapter with frozen-encoder caches and official fusion/loss."""
import argparse,hashlib,json,random,sys,types,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from huggingface_hub import hf_hub_download
ROOT=Path("runs/full_evaluation_20260921");EMPTY="No event is available at this forecast origin."
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);a=ap.parse_args();ds=a.dataset;out=ROOT/"baselines"/ds/"CAMEF_s2026";out.mkdir(parents=True,exist_ok=True)
 if (out/"COMPLETE").exists():return
 random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026);torch.set_num_threads(4)
 sys.path.insert(0,str(Path("vendor/CAMEF/CAMEF").resolve()));from model.CAMEF import CAMEF,contrastive_loss_objective
 paths={k:str(Path(hf_hub_download(repo,"config.json")).parent) for k,repo in [("bert","FacebookAI/roberta-base"),("moment","AutonLab/MOMENT-1-large"),("gpt","openai-community/gpt2")]}
 model=CAMEF(**paths,seq_len=96,pred_len=24,d=1,window=500,stride=400,batch_size=32);model.eval();data=ROOT/"data"/ds;idx=pd.read_csv(data/"index.csv");win=np.load(data/"windows.npz");x=win["x"][:,:,3:4].transpose(0,2,1).copy();y=win["y"].astype(np.float32);events=json.loads((data/"events.json").read_text())
 cut="2025-03-01" if ds=="celebrity" else "2025-09-01";train=np.flatnonzero((idx.split=="train")&(pd.to_datetime(idx.target_end,utc=True)<pd.Timestamp(cut,tz="UTC")));training=[]
 texts=["\n".join(e["text"] for e in group) if group else EMPTY for group in events];all_text=set(texts);cf={};neg={};typ={i:"+".join(sorted(set(e["topic"] for e in events[i]))) for i in train if events[i]};event_train=list(typ)
 for i in event_train:
  reports=[]
  for e in events[i]:
   record=json.loads((ROOT/"camef_counterfactuals"/(sha(e["text"])+".json")).read_text());assert record["status"]=="ok";reports.append(record["reports"])
  cf[i]=["\n".join(r[k] for r in reports) for k in range(10)];pool=[j for j in event_train if typ[j]!=typ[i]]
  if not pool:raise ValueError("No different-type training negatives")
  rng=np.random.default_rng(2026+i);neg[i]=[texts[j] for j in rng.choice(pool,size=5,replace=len(pool)<5)];all_text.update(cf[i]);all_text.update(neg[i])
 # News is inference only: use the celebrity-trained checkpoint, no news counterfactuals or targets are used for fitting.
 targets=[ds,"news"] if ds=="celebrity" else [ds];all_inputs={ds:(idx,x,y,texts)}
 if ds=="celebrity":
  nd=ROOT/"data/news";ni=pd.read_csv(nd/"index.csv");nw=np.load(nd/"windows.npz");ne=json.loads((nd/"events.json").read_text());nt=["\n".join(e["text"] for e in group) if group else EMPTY for group in ne];all_inputs["news"]=(ni,nw["x"][:,:,3:4].transpose(0,2,1).copy(),nw["y"],nt);all_text.update(nt)
 cache=ROOT/"camef_cache"/ds;cache.mkdir(parents=True,exist_ok=True);sorted_text=sorted(all_text);text_hash=sha(json.dumps(sorted_text,ensure_ascii=False));tp=cache/"text_embeddings.npz"
 if tp.exists():
  z=np.load(tp);assert str(z["hash"])==text_hash;emb=z["embeddings"]
 else:
  embeddings=[]
  with torch.no_grad():
   for start in range(0,len(sorted_text),32):
    batch=sorted_text[start:start+32];enc=model.tokenizer(batch,padding=True,truncation=False,return_tensors="pt")
    if enc.input_ids.shape[1]<=500:
     enc={k:v.cuda() for k,v in enc.items()};hidden=model.bert_model(**enc).last_hidden_state;mask=enc["attention_mask"].unsqueeze(-1);arr=(hidden*mask).sum(1)/mask.sum(1);embeddings.extend(arr.cpu().numpy())
    else:
     for text in batch:embeddings.append(model.text_embedding(text).cpu().numpy())
    if start%640==0:print("encoded text",start,"/",len(sorted_text),flush=True)
  emb=np.asarray(embeddings,dtype=np.float32);np.savez_compressed(tp,embeddings=emb,hash=np.asarray(text_hash))
 text_map={t:emb[i] for i,t in enumerate(sorted_text)};series={}
 with torch.no_grad():
  for name,(ix,xx,yy,tt) in all_inputs.items():
   path=cache/f"{name}_series.npy"
   if path.exists():series[name]=np.load(path);continue
   chunks=[]
   for lo in range(0,len(xx),32):
    inp=torch.from_numpy(xx[lo:lo+32]).cuda();mask=torch.zeros(len(inp),512,device="cuda",dtype=torch.long);mask[:,-96:]=1;vec=model.moment_model(x_enc=torch.nn.functional.pad(inp,(416,0)),input_mask=mask).embeddings;chunks.append(vec.cpu().numpy())
    if lo%3200==0:print("encoded series",name,lo,"/",len(xx),flush=True)
   series[name]=np.concatenate(chunks);np.save(path,series[name])
 # The original text_embedding and MOMENT forward are replaced only by their deterministic precomputed outputs.
 def cached_text(self,text):return torch.from_numpy(text_map[text]).cuda()
 class FrozenSeries(torch.nn.Module):
  def __init__(self):super().__init__();self.current=None
  def forward(self,x_enc=None,**kw):return SimpleNamespace(embeddings=self.current)
 model.text_embedding=types.MethodType(cached_text,model);model.bert_model=torch.nn.Identity();model.moment_model=FrozenSeries();torch.cuda.empty_cache()
 def forward_cached(batch_text,batch_seq,sent=None,negative=None):
  b=len(batch_text)
  def project(strings):return model.bert_linear(torch.from_numpy(np.stack([text_map[t] for t in strings])).cuda())
  text=project(batch_text);series_embedding=model.embed_project(model.moment_model.current)+model.residual_project(batch_seq.reshape(b,96))
  fused=model.fuse_project(torch.stack((text,series_embedding),dim=1).reshape(b,-1));hidden=model.gpt_model.drop(fused).unsqueeze(1)
  for layer in model.gpt_model.h:
   if isinstance(hidden,tuple):hidden=hidden[0]
   hidden=layer(hidden)
  hidden=model.gpt_model.ln_f(hidden[0]);output=model.output_project(hidden.reshape(b,768)).reshape(b,1,24)
  if sent is None:return output,hidden,series_embedding,text
  sent_emb=project([t for group in sent for t in group]).reshape(b,10,768);sent_emb=torch.cat([text.unsqueeze(1),sent_emb],dim=1);neg_emb=project([t for group in negative for t in group]).reshape(b,5,768)
  return output,hidden,series_embedding,text,sent_emb,neg_emb
 # Deterministic equivalence check against the upstream implementation before training.
 with torch.no_grad():
  ids=np.asarray(event_train[:2]);model.moment_model.current=torch.from_numpy(series[ds][ids]).cuda();sx=torch.from_numpy(x[ids]).cuda();st=[texts[i] for i in ids];sr=[cf[i] for i in ids];sn=[neg[i] for i in ids]
  original=model.predict_batch_contrastive(st,sr,sn,sx);batched=forward_cached(st,sx,sr,sn)
  errors=[float((a-b).abs().max()) for a,b in zip(original,batched)]
  for a,b in zip(original,batched):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
  (out/"vectorization_check.json").write_text(json.dumps(dict(passed=True,max_absolute_errors=errors,upstream="predict_batch_contrastive"),indent=2));print("Batched implementation matches upstream",errors,flush=True)

 for p in model.parameters():p.requires_grad=True
 for module in [model.gpt_model.wte,model.gpt_model.wpe]:
  for p in module.parameters():p.requires_grad=False
 groups=[(model.gpt_model.h,1e-5),(model.gpt_model.ln_f,1e-5),(model.fuse_project,5e-7),(model.bert_linear,5e-7),(model.embed_project,1e-5),(model.output_project,1e-5),(model.residual_project,1e-5)];optimizer=torch.optim.Adam([{"params":m.parameters(),"lr":lr} for m,lr in groups])
 import subprocess
 meta=dict(model="CAMEF",variant="Official fusion/loss with dataset adapter; deterministic frozen encoders",seed=2026,data=str(data.resolve()),weights=paths,upstream=str(Path("vendor/CAMEF").resolve()),commit=subprocess.check_output(["git","-C","vendor/CAMEF","rev-parse","HEAD"],text=True).strip(),epochs=20,n_train=len(train),n_event_train=len(event_train),train_before=cut,batch_size=32,input="96 close observations; MOMENT masked to512",counterfactual_prompt_sha256=sha((ROOT/"camef_counterfactuals/prompt.txt").read_text()),checkpoint_selection="fixed final epoch",source_code="reproduction/full_camef.py")
 (out/"provenance.json").write_text(json.dumps(meta,indent=2));history=[];start=time.time();first_epoch=0
 if (out/"resume.pt").exists():
  state=torch.load(out/"resume.pt",map_location="cuda",weights_only=False);model.load_state_dict(state["state_dict"]);optimizer.load_state_dict(state["optimizer"]);first_epoch=state["epoch"];history=state["history"]
 for epoch in range(first_epoch,20):
  model.train();rng=np.random.default_rng(2026+epoch);order=rng.permutation(train);total=0.
  for lo in range(0,len(order),32):
   ids=order[lo:lo+32];model.moment_model.current=torch.from_numpy(series[ds][ids]).cuda();xb=torch.from_numpy(x[ids]).cuda();optimizer.zero_grad(set_to_none=True)
   batch_text=[texts[i] for i in ids];br=[cf.get(i,[EMPTY]*10) for i in ids];bn=[neg.get(i,[EMPTY]*5) for i in ids]
   output,_,series_emb,_,reports_emb,negative_emb=forward_cached(batch_text,xb,br,bn);truth=torch.from_numpy(y[ids]).cuda().unsqueeze(1);loss=(output-truth).square().mean()+(output-truth).abs().mean();mask=torch.as_tensor([i in cf for i in ids],device="cuda")
   if mask.any():loss+=contrastive_loss_objective(series_emb[mask],reports_emb[mask],negative_emb[mask])
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ids)
  row=dict(epoch=epoch+1,loss=total/len(train),seconds=time.time()-start);history.append(row);print(json.dumps(row),flush=True);(out/"history.json").write_text(json.dumps(history,indent=2));torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,provenance=meta),out/"resume.pt")
 torch.save(dict(state_dict=model.state_dict(),provenance=meta),out/"model.pt");model.eval()
 with torch.no_grad():
  for name,(ix,xx,yy,tt) in all_inputs.items():
   dest=ROOT/"baselines"/name/"CAMEF_s2026";dest.mkdir(parents=True,exist_ok=True);pred=[]
   for lo in range(0,len(xx),64):
    model.moment_model.current=torch.from_numpy(series[name][lo:lo+64]).cuda();p=forward_cached(tt[lo:lo+64],torch.from_numpy(xx[lo:lo+64]).cuda())[0][:,0];pred.append(p.cpu().numpy())
   np.save(dest/"predictions.npy",np.concatenate(pred));(dest/"provenance.json").write_text(json.dumps(dict(meta,inference_dataset=name),indent=2));(dest/"COMPLETE").touch()
if __name__=="__main__":main()
