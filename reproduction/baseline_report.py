"""Report measured backbone errors without requiring event inputs."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
def scores(y, pred):
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return dict(
        mse=float(np.mean(error**2)),
        mae=float(np.mean(np.abs(error))),
        mape=float(np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-9)) * 100),
    )

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();data=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    index=pd.read_csv(data/'index.csv');y=np.load(data/'windows.npz')['y'];rows=[]
    for run in a.runs:
        run=Path(run)
        if not (run/'COMPLETE').exists():raise ValueError(f'Incomplete run: {run}')
        meta=json.loads((run/'provenance.json').read_text());pred=np.load(run/'predictions.npy')
        if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('Prediction/target mismatch')
        for asset in sorted(index.asset.unique()):
            ids=np.flatnonzero((index.asset==asset)&(index.split=='test'))
            for h in [0,6,12,24]:
                truth=y[ids] if h==0 else y[ids,h-1]
                forecast=pred[ids] if h==0 else pred[ids,h-1]
                rows.append(dict(model=meta['model'],seed=meta['seed'],asset=asset,horizon='avg' if h==0 else h,n=len(ids),**scores(truth,forecast)))
    result=pd.DataFrame(rows);result.to_csv(out/'baseline_metrics.csv',index=False)
    (out/'README.md').write_text('New measured backbone runs. No NSRC correction or event-conditioned significance is claimed. MSE and MAE use origin-price-normalized values; MAPE is in percent. Historical manuscript results are not overwritten.\n')
    print(result[result.horizon=='avg'].to_string(index=False))
if __name__=='__main__':main()
