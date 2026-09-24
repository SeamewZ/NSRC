"""Run an existing checkpoint on a revised, explicitly indexed evaluation set."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from reproduction.train import build

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--vendor',default='vendor');p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args();data=Path(a.data);checkpoint=Path(a.checkpoint);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=False);meta=state['provenance']
    model,forward,source=build(meta['model'],Path(a.vendor).resolve(),device)
    if source['commit']!=meta['commit']:raise ValueError('Upstream revision changed')
    model.load_state_dict(state['state_dict']);model.eval();x=np.load(data/'windows.npz')['x'];pred=[]
    with torch.no_grad():
        for lo in range(0,len(x),a.batch_size):pred.append(forward(torch.from_numpy(x[lo:lo+a.batch_size]).to(device)).cpu().numpy())
    np.save(out/'predictions.npy',np.concatenate(pred))
    meta.update(inference_data=str(data.resolve()),checkpoint=str(checkpoint.resolve()),index_sha256=hashlib.sha256((data/'index.csv').read_bytes()).hexdigest())
    (out/'provenance.json').write_text(json.dumps(meta,indent=2));(out/'COMPLETE').write_text('Checkpoint inference complete\n');print('COMPLETE',str(out),flush=True)
if __name__=='__main__':main()
