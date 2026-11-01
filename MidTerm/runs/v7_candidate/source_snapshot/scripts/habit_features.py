"""Programme habit persistence and recent rate changes from past events only."""
import datetime as dt
import numpy as np

NAMES=['habit_log_active_days','habit_log_active_weeks','habit_week_coverage','habit_log_events_per_active_day',
       'habit_recent14_share','habit_recent14_per_air','habit_previous14_per_air','habit_rate_trend',
       'habit_global_recent_audience','habit_global_previous_audience','habit_global_audience_trend']


def augment_habit(data,con,start,cutoff,targets):
    begin=dt.datetime.fromisoformat(start);cut=dt.datetime.fromisoformat(cutoff)
    recent=max(begin,cut-dt.timedelta(days=14));previous=max(begin,cut-dt.timedelta(days=28))
    slot_recent=f"air_start>=TIMESTAMP '{recent}'"
    slot_previous=f"air_start>=TIMESTAMP '{previous}' AND air_end<=TIMESTAMP '{recent}'"
    rows=con.execute(f'''SELECT user_id,item,count(*),count(DISTINCT date(air_start)),count(DISTINCT date_trunc('week',air_start)),
      count(*) FILTER(WHERE {slot_recent}),count(*) FILTER(WHERE {slot_previous})
      FROM air_views WHERE item>0 AND fraction>=.8 AND air_start>='{start}' AND air_end<='{cutoff}' GROUP BY 1,2''').fetchall()
    lookup={(u,int(i)):(n,days,weeks,n14,p14) for u,i,n,days,weeks,n14,p14 in rows}
    global_rows=con.execute(f'''SELECT item,count(DISTINCT user_id) FILTER(WHERE {slot_recent}),count(DISTINCT user_id) FILTER(WHERE {slot_previous})
      FROM air_views WHERE item>0 AND fraction>=.8 AND air_start>='{start}' AND air_end<='{cutoff}' GROUP BY 1''').fetchall()
    audience={int(i):(a,b) for i,a,b in global_rows}
    supply={int(i):(a,b) for i,a,b in con.execute(f'''SELECT item,
      count(*) FILTER(WHERE s>='{recent}'),count(*) FILTER(WHERE s>='{previous}' AND e<='{recent}')
      FROM schedule WHERE item>0 AND s>='{start}' AND e<='{cutoff}' GROUP BY 1''').fetchall()}
    total_weeks=max(1,(cut-begin).total_seconds()/86400/7)
    features=[]
    for r,c in zip(data['rr'],data['cc']):
        item=int(data['items'][c]);user=targets[r]
        n,days,weeks,n14,p14=lookup.get((user,item),(0,0,0,0,0))
        a,b=supply.get(item,(0,0));g14,gp14=audience.get(item,(0,0))
        rate=n14/max(a,1);old_rate=p14/max(b,1)
        features.append([np.log1p(days),np.log1p(weeks),weeks/total_weeks,np.log1p(n/max(days,1)),
          n14/max(n,1),rate,old_rate,np.log((rate+.01)/(old_rate+.01)),
          np.log1p(g14),np.log1p(gp14),np.log((g14+5)/(gp14+5))])
    extra=np.asarray(features,dtype=np.float32)
    assert np.isfinite(extra).all()
    return {**data,'x':np.column_stack([data['x'],extra]).astype(np.float32),'feature_names':np.concatenate([data['feature_names'],np.array(NAMES)])}
