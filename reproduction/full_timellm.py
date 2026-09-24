"""Official Time-LLM GPT2-6 adaptation; architecture code is not rewritten."""
import argparse,json,random,sys,time,subprocess,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path("runs/full_evaluation_20260921")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True);ap.add_argument("--seed",type=int,default=2026);a=ap.parse_args();ds=a.dataset
    out=ROOT/"baselines"/ds/f"Time-LLM_s{a.seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE").exists():return
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(4)
    upstream=Path("vendor/Time-LLM").resolve();sys.path.insert(0,str(upstream));from models.TimeLLM import Model
    cfg=SimpleNamespace(task_name="long_term_forecast",pred_len=24,seq_len=96,d_ff=32,llm_dim=768,patch_len=16,stride=8,llm_model="GPT2",llm_layers=6,prompt_domain=1,content="Hourly OHLCV observations of financial assets. Forecast future observations using only the past 96 observations.",d_model=32,n_heads=8,enc_in=1,dropout=.1)
    model=Model(cfg).cuda();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    data=Path("runs/celebrity_validation_v1/data") if ds=="celebrity" else Path("runs/icassp_reproduction")/ds
    win=np.load(data/"windows.npz");x=win["x"][:,:,3:4].copy();y=win["y"];idx=pd.read_csv(data/"index.csv");train=(idx.split=="train").to_numpy()
    if ds=="celebrity":train&=(pd.to_datetime(idx.target_end,utc=True)<pd.Timestamp("2025-03-01",tz="UTC")).to_numpy()
    meta=dict(model="Time-LLM",variant="Official GPT2 backbone, first 6 layers, close-only channel-independent input",seed=a.seed,config=vars(cfg),upstream=str(upstream),commit=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip(),data=str(data.resolve()),epochs=20,batch_size=32,lr=1e-4,n_train=int(train.sum()),train_before="2025-03-01" if ds=="celebrity" else "2025-09-01",checkpoint_selection="fixed final epoch",precision="CUDA autocast bfloat16; official hardcoded bfloat16 patch input retained")
    (out/"provenance.json").write_text(json.dumps(meta,indent=2));loader=DataLoader(TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y[train])),batch_size=32,shuffle=True);history=[];start=time.time()
    for epoch in range(20):
        model.train();loss_sum=0
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):p=model(xb.cuda(),None,None,None)[:,:,0];loss=(p.float()-yb.cuda()).abs().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=loss.item()*len(xb)
        record=dict(epoch=epoch+1,mae=loss_sum/train.sum(),seconds=time.time()-start);history.append(record);print(json.dumps(record),flush=True);(out/"history.json").write_text(json.dumps(history,indent=2))
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,provenance=meta),out/"resume.pt")
    torch.save(dict(state_dict=model.state_dict(),provenance=meta),out/"model.pt")
    model.eval()
    for target in ([ds,"news"] if ds=="celebrity" else [ds]):
        dest=ROOT/"baselines"/target/f"Time-LLM_s{a.seed}";dest.mkdir(parents=True,exist_ok=True);tx=np.load(ROOT/"data"/target/"windows.npz")["x"][:,:,3:4].copy();pred=[]
        with torch.no_grad():
            for lo in range(0,len(tx),32):
                with torch.autocast("cuda",dtype=torch.bfloat16):p=model(torch.from_numpy(tx[lo:lo+32]).cuda(),None,None,None)[:,:,0]
                pred.append(p.float().cpu().numpy())
        np.save(dest/"predictions.npy",np.concatenate(pred));(dest/"provenance.json").write_text(json.dumps(dict(meta,inference_dataset=target),indent=2));(dest/"COMPLETE").write_text("Official Time-LLM GPT2-6 training and inference complete\n")
if __name__=="__main__":main()
