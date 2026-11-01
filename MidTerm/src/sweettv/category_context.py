"""Soft category/time preferences for mixed-use accounts; no viewer identity claim.

Categories come only from the model's pre-cutoff metadata. Unknown future item
categories remain unknown, even when the future schedule contains a label.
"""
import numpy as np
from scipy import sparse


def category_time_features(con,model,start,cutoff,end,sigma_hours=3,prior_events=5,half_life_days=28):
    assert sigma_hours>0 and prior_events>0 and half_life_days>0
    categories=sorted({r['tv_show_category'] for r in model.metadata.values() if r.get('tv_show_category')})
    shape=(len(model.target_users),len(model.candidates))
    names=['category_known','category_history_share','category_recent_share','category_recent_trend','category_clock_affinity','category_clock_lift']
    if not categories:return {name:np.zeros(shape) for name in names}
    catix={cat:i for i,cat in enumerate(categories)}
    itemcat={int(item):catix[r['tv_show_category']] for item,r in model.metadata.items() if r.get('tv_show_category')}
    # Only the already-filtered metadata is materialized. Future schedule labels
    # are never queried: its ID and timing suffice to project known categories.
    values=','.join(f'({item},{cat})' for item,cat in sorted(itemcat.items()))
    con.execute(f'CREATE OR REPLACE TEMP TABLE category_time_map AS SELECT * FROM (VALUES {values}) supplied(item,category_id)')
    tix={u:i for i,u in enumerate(model.target_users)};nc=len(categories);nslots=48
    full=np.zeros((len(tix),nc,nslots));recent=np.zeros_like(full)
    rows=con.execute(f'''WITH expanded AS (
      SELECT v.*,m.category_id,h.hh FROM air_views v JOIN category_time_map m USING(item),
        LATERAL generate_series(date_trunc('hour',air_start),date_trunc('hour',air_end-INTERVAL '1 microsecond'),INTERVAL '1 hour') h(hh)
      WHERE fraction>=.8 AND air_start>='{start}' AND air_end<='{cutoff}')
      SELECT user_id,category_id,CASE WHEN dayofweek(hh) IN(0,6) THEN 24 ELSE 0 END+hour(hh) slot,
        sum(epoch(least(air_end,hh+INTERVAL '1 hour')-greatest(air_start,hh))/duration) full_events,
        sum(epoch(least(air_end,hh+INTERVAL '1 hour')-greatest(air_start,hh))/duration*
          pow(2.0,-epoch(TIMESTAMP '{cutoff}'-air_end)/86400/{half_life_days})) recent_events
      FROM expanded GROUP BY 1,2,3''').fetchall()
    for user,cat,slot,a,b in rows:
        if user in tix:full[tix[user],cat,slot]=a;recent[tix[user],cat,slot]=b
    def shares(hist):
        sums=hist.sum(axis=2)
        return np.divide(sums,sums.sum(axis=1,keepdims=True),out=np.zeros_like(sums),where=sums.sum(axis=1,keepdims=True)>0)
    full_share=shares(full);recent_share=shares(recent)
    distance=np.minimum(np.arange(24),24-np.arange(24))
    kernel=np.exp(-.5*(distance/sigma_hours)**2);kernel/=kernel.sum()
    source=recent.reshape(len(tix),nc,2,24);smooth=np.zeros_like(source)
    for shift,weight in enumerate(kernel):smooth+=weight*np.roll(source,shift,axis=-1)
    smooth=smooth.reshape(recent.shape)
    conditional=(smooth+prior_events*recent_share[:,:,None])/(smooth.sum(axis=1,keepdims=True)+prior_events)
    rr,cc,vv=[],[],[];supply=np.zeros(len(model.candidates));candidate_ix={int(item):j for j,item in enumerate(model.candidates)}
    query=f'''WITH expanded AS (
      SELECT a.item,a.s,a.e,a.duration,m.category_id,h.hh FROM schedule a JOIN category_time_map m USING(item),
        LATERAL generate_series(date_trunc('hour',s),date_trunc('hour',e-INTERVAL '1 microsecond'),INTERVAL '1 hour') h(hh)
      WHERE item>0 AND s>='{cutoff}' AND e<='{end}')
      SELECT category_id,CASE WHEN dayofweek(hh) IN(0,6) THEN 24 ELSE 0 END+hour(hh) slot,item,
        sum(epoch(least(e,hh+INTERVAL '1 hour')-greatest(s,hh))/duration) exposure
      FROM expanded GROUP BY 1,2,3'''
    for cat,slot,item,value in con.execute(query).fetchall():
        if item in candidate_ix:
            j=candidate_ix[item];rr.append(cat*nslots+slot);cc.append(j);vv.append(value);supply[j]+=value
    exposure=sparse.csr_matrix((vv,(rr,cc)),shape=(nc*nslots,len(model.candidates)))
    affinity=np.asarray(conditional.reshape(len(tix),-1)@exposure)/np.maximum(supply,1)[None,:]
    history_share=np.zeros(shape);new_share=np.zeros(shape);known=np.zeros(shape)
    for j,item in enumerate(model.candidates):
        if int(item) in itemcat:
            cat=itemcat[int(item)];known[:,j]=1;history_share[:,j]=full_share[:,cat];new_share[:,j]=recent_share[:,cat]
    return {'category_known':known,'category_history_share':history_share,'category_recent_share':new_share,
            'category_recent_trend':np.where(known>0,(new_share+.01)/(history_share+.01),0),
            'category_clock_affinity':affinity,'category_clock_lift':np.where(known>0,affinity/(new_share+.01),0)}
