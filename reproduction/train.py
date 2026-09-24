"""Train official backbones on prepared windows; save all split predictions."""
import argparse,importlib,json,os,random,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset


def build(name,root,device):
    if name=='GPT4TS':
        path=root/'OneFitsAll'/'Long-term_Forecasting'
        sys.path.insert(0,str(path));sys.path.insert(0,str(path/'models'))
        cls=importlib.import_module('GPT4TS').GPT4TS
        cfg=SimpleNamespace(seq_len=96,pred_len=24,patch_size=16,stride=8,d_model=768,gpt_layers=6,pretrain=1,freeze=1,is_gpt=1)
        model=cls(cfg,device);forward=lambda x:model(x,0)[:,:,3]
        upstream=root/'OneFitsAll'
    else:
        upstream=root/'Time-Series-Library';sys.path.insert(0,str(upstream))
        cls=importlib.import_module('models.'+name).Model
        cfg=SimpleNamespace(task_name='long_term_forecast',seq_len=96,pred_len=24,enc_in=5,c_out=5,
              moving_avg=25,d_model=128,n_heads=4,e_layers=2,d_ff=256,dropout=.1,factor=5,
              activation='gelu',embed='timeF',freq='h',output_attention=False,use_norm=True,label_len=48,top_k=5,num_kernels=6)
        model=cls(cfg).to(device);forward=lambda x:model(x,None,None,None)[:,:,3]
    rev=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    return model,forward,dict(config=vars(cfg),upstream=str(upstream),commit=rev)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model',choices=['GPT4TS','DLinear','PatchTST','iTransformer','TimesNet'],required=True)
    p.add_argument('--vendor',default='vendor');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--train-before',default=None);p.add_argument('--seed',type=int,default=2026);p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=np.load(Path(a.data)/'windows.npz');x=ds['x'];y=ds['y'];index=pd.read_csv(Path(a.data)/'index.csv')
    train=np.flatnonzero(index.split=='train');val=np.flatnonzero(index.split=='val')
    if a.train_before:train=train[(pd.to_datetime(index.iloc[train].target_end,utc=True)<pd.Timestamp(a.train_before,tz='UTC')).to_numpy()]
    assert len(train)>0 and len(val)>0
    model,forward,source=build(a.model,Path(a.vendor).resolve(),device)
    source.update(train_before=a.train_before,n_train=len(train),model=a.model,seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,lr=.0001 if a.model=='GPT4TS' else .001,
        loss='MAE on origin-price-normalized close',checkpoint_selection='fixed final epoch; calibration validation not used for early stopping',
        training='joint model across assets in the input dataset, recorded reconstruction choice',data=str(Path(a.data).resolve()),torch_version=torch.__version__,device=str(device))
    (out/'provenance.json').write_text(json.dumps(source,indent=2))
    loader=DataLoader(TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y[train])),batch_size=a.batch_size,shuffle=True)
    optim=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=source['lr'],weight_decay=1e-4)
    history=[];start=time.time()
    for epoch in range(a.epochs):
        model.train();loss_sum=0
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device);optim.zero_grad(set_to_none=True)
            pred=forward(xb);loss=(pred-yb).abs().mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step();loss_sum+=loss.item()*len(xb)
        row=dict(epoch=epoch+1,mae=loss_sum/len(train),seconds=time.time()-start);history.append(row)
        print(json.dumps(row),flush=True);(out/'history.json').write_text(json.dumps(history,indent=2))
    torch.save(dict(state_dict=model.state_dict(),provenance=source),out/'model.pt')
    model.eval();preds=[]
    with torch.no_grad():
        for lo in range(0,len(x),a.batch_size):preds.append(forward(torch.from_numpy(x[lo:lo+a.batch_size]).to(device)).cpu().numpy())
    np.save(out/'predictions.npy',np.concatenate(preds));(out/'COMPLETE').write_text('training and all-split inference complete\n')
    print('COMPLETE',a.out,flush=True)

if __name__=='__main__':main()
