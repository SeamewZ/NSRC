"""Prepare new baseline runs from complete server prices while event data are pending."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

FILES={'BTC':'Bitcoin(BTC)_21_25.csv','ETH':'Ethereum_21_25.csv','SOL':'Solana_21_25.csv','DOGE':'Dogecoin_21_25.csv',
       'SPX':'SPX-60min_utc.csv','NDX':'NDX-60min_utc.csv','INDU':'INDU-60min_utc.csv'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True)
    p.add_argument('--assets',nargs='+',default=['BTC','ETH','SOL','DOGE']);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);xs=[];ys=[];rows=[];provenance=[]
    boundaries=[pd.Timestamp(s,tz='UTC') for s in ['2025-01-01','2025-09-01','2025-11-01','2025-12-31']]
    for asset in a.assets:
        f=Path(a.raw)/FILES[asset];df=pd.read_csv(f).rename(columns={'date':'timestamp','open_time':'timestamp'})
        df['timestamp']=pd.to_datetime(df.timestamp,utc=True);df=df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        if 'volume' not in df:df['volume']=0.
        df=df[(df.timestamp>=boundaries[0]-pd.Timedelta(days=35))&(df.timestamp<boundaries[-1])].reset_index(drop=True)
        prices=df[['open','high','low','close','volume']].to_numpy(float);times=df.timestamp.to_numpy()
        for i in range(95,len(df)-24):
            t=df.timestamp.iloc[i];target_end=df.timestamp.iloc[i+24]
            if t<boundaries[0]:continue
            if asset in ['BTC','ETH','SOL','DOGE'] and ((np.diff(times[i-95:i+25])!=pd.Timedelta(hours=1)).any()):continue
            split='train' if t<boundaries[1] else ('val' if t<boundaries[2] else 'test');boundary={'train':boundaries[1],'val':boundaries[2],'test':boundaries[3]}[split]
            if target_end>=boundary:continue
            close=max(abs(prices[i,3]),1e-9);x=prices[i-95:i+1].copy();x[:,:4]/=close;x[:,4]/=max(x[:,4].mean(),1e-9)
            xs.append(x);ys.append(prices[i+1:i+25,3]/close)
            rows.append(dict(asset=asset,timestamp=str(t),target_end=str(target_end),split=split,origin_close=close,events=0))
        provenance.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),rows=len(df)))
    idx=pd.DataFrame(rows);idx.to_csv(out/'index.csv',index=False);np.savez_compressed(out/'windows.npz',x=np.array(xs,dtype='float32'),y=np.array(ys,dtype='float32'))
    (out/'events.json').write_text(json.dumps([[] for _ in rows]))
    meta=dict(status='BASELINE_ONLY_EVENT_DATA_PENDING',protocol='2025 Jan-Aug train / Sep-Oct validation / Nov-Dec test; L96 H24',
      inputs='origin-scaled OHLC, history-mean-scaled volume',purge='discard target windows crossing split boundary',
      index_horizon='24 observed trading bars' if 'SPX' in a.assets else '24 contiguous hourly bars',
      files=provenance,split_counts={f'{asset}/{s}':int(n) for (asset,s),n in idx.groupby(['asset','split']).size().items()})
    (out/'data_audit.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
