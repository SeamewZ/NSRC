"""Prepare auditable windows. Never silently substitute missing test dates."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from celebcrypto.data.bundled import load_bundled_market,load_bundled_tweets
from celebcrypto.data.preprocess import aggregate_5m_to_hourly


def prepare(bundle, out, protocol='paper', description_field='key_word_used'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=load_bundled_market(bundle);tweets=load_bundled_tweets(bundle)
    assets=['BTC','ETH','SOL','DOGE'];raw=raw[raw.asset.isin(assets)]
    tweets=tweets[tweets.asset.isin(assets)&tweets.timestamp.notna()].copy()
    # The public snapshot supplies one label per text, not individual model votes.
    # Preserve it and its provenance; do not fabricate an ensemble vote record.
    tweets=tweets.drop_duplicates(['asset','timestamp','author_id','text'])
    audit=dict(protocol=protocol,market_min=str(raw.timestamp.min()),market_max=str(raw.timestamp.max()),
               event_min=str(tweets.timestamp.min()),event_max=str(tweets.timestamp.max()),
               source='bundled recorded labels (individual votes unavailable)',description_field=description_field)
    if description_field not in ['key_word_used','reasoning','text']:
        raise ValueError('Choose an existing event-description field, or provide an explicit input adapter.')
    audit['description_note']='Reconstruction uses the chosen recorded field verbatim as z; not confirmed as original experimental z.'
    audit['events_per_month']=tweets.groupby(tweets.timestamp.dt.strftime('%Y-%m')).size().to_dict()
    audit['ohlcv_per_asset']=raw.groupby('asset').agg(start=('timestamp','min'),end=('timestamp','max'),rows=('timestamp','size')).astype(str).to_dict('index')
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    if protocol=='paper':
        train_end=pd.Timestamp('2025-09-01',tz='UTC');val_end=pd.Timestamp('2025-11-01',tz='UTC');end=pd.Timestamp('2025-12-31',tz='UTC')
    else:
        train_end=pd.Timestamp('2025-04-01',tz='UTC');val_end=pd.Timestamp('2025-05-01',tz='UTC');end=pd.Timestamp('2025-06-01',tz='UTC')
    if raw.timestamp.max()<end-pd.Timedelta(days=1) or tweets.timestamp.max()<end-pd.Timedelta(days=2):
        raise ValueError(f'Missing market/event coverage for {protocol}: see {out}/data_audit.json; cannot reproduce these splits.')
    hourly=aggregate_5m_to_hourly(raw)
    tweets=tweets[(tweets.timestamp>=pd.Timestamp('2025-01-01',tz='UTC'))&(tweets.timestamp<end)].copy()
    tweets['engagement']=tweets[['like_count','retweet_count','reply_count','quote_count']].fillna(0).sum(axis=1)
    # Mirror the release's daily top-decile inclusion; this is retrospective dataset
    # construction, not a claim that collection-time engagement was known at origin.
    cutoff=tweets.groupby(tweets.timestamp.dt.date).engagement.transform(lambda s:s.quantile(.9))
    tweets=tweets[tweets.engagement>=cutoff].copy()
    tweets['origin']=tweets.timestamp.dt.floor('h')-pd.Timedelta(hours=1)
    records={}
    for row in tweets.itertuples():
        z=str(getattr(row,description_field) or '').strip()
        if not z:continue
        event=dict(c='CelebrityTweet',d=str(row.label),z=z,event_id=str(row.tweet_id),published=str(row.timestamp))
        records.setdefault((row.asset,row.origin),[]).append(event)
    xs=[];ys=[];rows=[];events=[]
    for asset,df in hourly.groupby('asset'):
        df=df.sort_values('timestamp').reset_index(drop=True)
        df=df[(df.timestamp>=pd.Timestamp('2025-01-01',tz='UTC'))&(df.timestamp<end)].reset_index(drop=True)
        times=df.timestamp.tolist();values=df[['open','high','low','close','volume']].to_numpy(float)
        for i in range(95,len(df)-24):
            t=times[i];future=times[i+24]
            if any(times[j+1]-times[j]!=pd.Timedelta(hours=1) for j in range(i-95,i+24)):continue
            split='train' if t<train_end else ('val' if t<val_end else 'test')
            boundary=train_end if split=='train' else (val_end if split=='val' else end)
            if future>=boundary:continue
            price=max(abs(values[i,3]),1e-9);x=values[i-95:i+1].copy()
            x[:,:4]/=price;x[:,4]/=max(x[:,4].mean(),1e-9)
            y=values[i+1:i+25,3]/price
            xs.append(x);ys.append(y);events.append(records.get((asset,t),[]))
            rows.append(dict(asset=asset,timestamp=str(t),split=split,origin_close=price,target_end=str(future),events=len(events[-1])))
    index=pd.DataFrame(rows);index.to_csv(out/'index.csv',index=False)
    np.savez_compressed(out/'windows.npz',x=np.asarray(xs,dtype='float32'),y=np.asarray(ys,dtype='float32'))
    (out/'events.json').write_text(json.dumps(events))
    audit['split_counts']=index.groupby(['asset','split']).size().to_dict()
    audit['split_counts']={f'{a}/{s}':int(n) for (a,s),n in audit['split_counts'].items()}
    audit['event_origins']=int((index.events>0).sum());audit['filter']='daily top-decile engagement in archived snapshot'
    audit['category_note']='All bundled tweets use CelebrityTweet. Category diversity is absent in this reconstruction.'
    audit['input_hash']=hashlib.sha256((out/'index.csv').read_bytes()+(out/'events.json').read_bytes()).hexdigest()
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',default='assets/dataset');p.add_argument('--out',required=True)
    p.add_argument('--protocol',choices=['paper','reconstruction'],default='paper');p.add_argument('--description-field',default='key_word_used')
    a=p.parse_args();prepare(a.bundle,a.out,a.protocol,a.description_field)
