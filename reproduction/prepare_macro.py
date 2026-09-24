"""Event-only, deterministic macro adaptation of symbolic residual memory."""
import argparse,hashlib,json,re
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

MONTHS='Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Q[1-4]'
def canonical_title(title):
    return re.sub(r'\s+',' ',re.sub(r'\s*\((?:'+MONTHS+r')\)\s*$','',title)).strip()

def number(s):
    s=str(s).strip().replace(',','').replace('−','-')
    m=re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([KMB%]?)',s,re.I)
    if not m:return None
    return float(m[1])*{'K':1e3,'M':1e6,'B':1e9,'%':.01,'':1}[m[2].upper()]

def compare(a,b):
    if a is None or b is None:return 'Unavailable'
    if np.isclose(a,b,rtol=1e-9,atol=1e-12):return 'Equal'
    return 'Above' if a>b else 'Below'

def symbols(row):
    # Directions denote released numeric surprise, NOT unobserved market direction.
    c=canonical_title(row['event']);a=number(row.get('actual',''))
    surprise=compare(a,number(row.get('forecast','')))
    trend=compare(a,number(row.get('previous','')))
    return dict(c=c,d=surprise,z=f'{c}; actual_vs_forecast={surprise}; actual_vs_previous={trend}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--events',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();data=Path(a.data).resolve();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    index=pd.read_csv(data/'index.csv');all_events=json.loads(Path(a.events).read_text());records=[];skips=Counter()
    for row in all_events:
        if row.get('event')=='Unknown':skips['unknown']+=1;continue
        if row.get('time')=='All Day':skips['no_precise_release_time']+=1;continue
        try:released=pd.Timestamp(datetime.strptime(row['date']+' '+row['time'],'%A, %B %d, %Y %H:%M'),tz='UTC')
        except (ValueError,KeyError):skips['unparseable_time']+=1;continue
        if released.year!=2025:continue
        eid=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        records.append(dict(**symbols(row),event_id=eid,published=str(released),boundary=released.floor('h')-pd.Timedelta(hours=1)))
    assigned=[[] for _ in range(len(index))];seen=set();skipped_alignment=Counter()
    # Use full numerical origin grid to avoid shifting events across purged boundaries.
    for asset,group in index.groupby('asset',sort=False):
        times=pd.DatetimeIndex(pd.to_datetime(group.timestamp,utc=True));ids=group.index.to_numpy();is_equity=asset in ['SPX','NDX','INDU']
        for r in records:
            boundary=r['boundary'];pos=times.searchsorted(boundary,side='right')-1
            if pos<0:skipped_alignment[asset]+=1;continue
            origin=times[pos];i=int(ids[pos])
            if (not is_equity and origin!=boundary) or (is_equity and boundary-origin>pd.Timedelta(days=4)):
                skipped_alignment[asset]+=1;continue
            released=pd.Timestamp(r['published']);split=index.iloc[i]['split'];expected='train' if released<pd.Timestamp('2025-09-01',tz='UTC') else ('val' if released<pd.Timestamp('2025-11-01',tz='UTC') else 'test')
            if split!=expected or released>=pd.Timestamp(index.iloc[i]['target_end']):skipped_alignment[asset]+=1;continue
            key=(asset,r['event_id'])
            if key in seen:continue
            seen.add(key);assigned[i].append({k:v for k,v in r.items() if k!='boundary'})
    index['events']=[len(e) for e in assigned];index.to_csv(out/'index.csv',index=False)
    f=out/'windows.npz'
    if not f.exists():f.symlink_to(data/'windows.npz')
    (out/'events.json').write_text(json.dumps(assigned,indent=2))
    audit=json.loads((data/'data_audit.json').read_text());audit.update(status='MACRO_ADAPTATION',event_source=str(Path(a.events).resolve()),event_sha256=hashlib.sha256(Path(a.events).read_bytes()).hexdigest(),
      event_encoding='Deterministic adaptation, not the original three-LLM celebrity labels: c=event title without release-month suffix; d=actual vs forecast (Above/Below/Equal/Unavailable); z=canonical title plus actual-vs-forecast and actual-vs-previous description. Exact matching. No target-derived teacher fields used.',
      timing='Reuse R2R clock values as UTC, floor release hour minus one; equities use preceding available observation. This is post-release correction of a pre-release baseline. Raw source lacks explicit timezone metadata.',
      event_rows_2025=len(records),event_skipped=dict(skips),alignment_skipped=dict(skipped_alignment),
      event_origins={f'{asset}/{split}':int((g.events>0).sum()) for (asset,split),g in index.groupby(['asset','split'])},
      event_instances={f'{asset}/{split}':int(g.events.sum()) for (asset,split),g in index.groupby(['asset','split'])})
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:audit[k] for k in ['status','event_rows_2025','event_origins','event_instances']},indent=2))
if __name__=='__main__':main()
