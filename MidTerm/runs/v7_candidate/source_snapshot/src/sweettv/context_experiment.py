"""Additional context experiments, with the same three rolling protocols."""
import datetime as dt
import json
import time
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .advanced import channel_prior, cosine_scores, item_supply
from .experiment import candidates_for,evaluate,load_metadata,make_truth,train_rows
from .models import Models,row_max_norm
from .prepare import dump


def time_context(con,model,start,cutoff,end,shrink_days=7,anchor='start',bucket_hours=4,day_mode='weekend',smooth_sigma_hours=0,half_life_days=None):
    """Raw viewing context: channel × day type × clock block.

    Context may use unlabelled airtime; item interactions still require ID>0, 80%.
    Exact timestamps are clipped at the train boundary before hourly expansion.
    """
    assert bucket_hours in (1,2,4,6,12,24)
    assert day_mode in ('weekend','weekday')
    assert half_life_days in (None,28)
    assert smooth_sigma_hours>=0
    per_day=24//bucket_hours
    def slot_sql(stamp):
        day=f"CASE WHEN dayofweek({stamp}) IN (0,6) THEN 1 ELSE 0 END" if day_mode=='weekend' else f'dayofweek({stamp})'
        return f'({day})*{per_day}+floor(hour({stamp})/{bucket_hours})::INTEGER'
    def recency(stamp):
        return '1.0::DOUBLE' if half_life_days is None else f"pow(2.0,-epoch(TIMESTAMP '{cutoff}'-({stamp}))/86400/{half_life_days})"
    channels=sorted(x[0] for x in con.execute('SELECT DISTINCT channel_id FROM schedule').fetchall())
    ci={c:i for i,c in enumerate(channels)};tix={u:i for i,u in enumerate(model.target_users)}
    nslots=per_day*(2 if day_mode=='weekend' else 7);nc=len(channels)
    hist=np.zeros((len(tix),nc*nslots))
    rows=con.execute(f'''WITH clipped AS (
      SELECT user_id,channel_id,greatest(s,TIMESTAMP '{start}') s,least(e,TIMESTAMP '{cutoff}') e
      FROM logs WHERE s<'{cutoff}' AND e>'{start}'),
      hours AS(SELECT l.*,h.hh FROM clipped l,LATERAL generate_series(date_trunc('hour',s),date_trunc('hour',e-INTERVAL '1 microsecond'),INTERVAL '1 hour') h(hh))
      SELECT user_id,channel_id,{slot_sql('hh')} slot,
      sum(epoch(least(e,hh+INTERVAL '1 hour')-greatest(s,hh))*{recency("hh+INTERVAL '30 minutes'")}) watched_seconds
      FROM hours GROUP BY 1,2,3''').fetchall()
    for u,ch,slot,seconds in rows:
        if u in tix and ch in ci:hist[tix[u],ci[ch]*nslots+slot]=seconds
    denominator=np.zeros(nslots)
    for slot,n in con.execute(f'''SELECT {slot_sql('hh')} slot,
       sum(3600*{recency("hh+INTERVAL '30 minutes'")}) FROM generate_series(TIMESTAMP '{start}',TIMESTAMP '{cutoff}'-INTERVAL '1 hour',INTERVAL '1 hour') h(hh) GROUP BY 1''').fetchall():denominator[slot]=n
    if smooth_sigma_hours:
        # Gaussian distance on the clock: close hours contribute continuously.
        # Seven-day profiles wrap over 168h; weekend/weekday profiles each wrap
        # over 24h, keeping their separate day types. Apply the same kernel to
        # numerator and clock exposure, so missing slots do not bias the rate.
        period=nslots if day_mode=='weekday' else per_day
        distance=np.minimum(np.arange(period),period-np.arange(period))*bucket_hours
        kernel=np.exp(-.5*(distance/smooth_sigma_hours)**2);kernel/=kernel.sum()
        shape=(len(tix),nc,1 if day_mode=='weekday' else 2,period)
        source=hist.reshape(shape);smoothed=np.zeros_like(source)
        den_source=denominator.reshape(-1,period);den_smoothed=np.zeros_like(den_source)
        for shift,weight in enumerate(kernel):
            if weight>1e-8:
                smoothed+=weight*np.roll(source,shift,axis=-1)
                den_smoothed+=weight*np.roll(den_source,shift,axis=-1)
        hist=smoothed.reshape(hist.shape);denominator=den_smoothed.ravel()
    base=hist.reshape(len(tix),nc,nslots).sum(axis=2)/denominator.sum()
    # Shrink each context rate toward the user's channel-wide time occupancy.
    tau=shrink_days*bucket_hours*3600
    rate=(hist+tau*np.repeat(base,nslots,axis=1))/(np.tile(denominator,nc)[None,:]+tau)
    rate=np.clip(rate,0,1)
    def exposures(left,right):
        rr,cc,vv=[],[],[]
        airing_weight=recency('e') if right==cutoff else '1.0'
        if anchor=='mean':
            query=f'''WITH hours AS(SELECT a.*,h.hh FROM schedule a,
              LATERAL generate_series(date_trunc('hour',s),date_trunc('hour',e-INTERVAL '1 microsecond'),INTERVAL '1 hour') h(hh)
              WHERE item>0 AND s>='{left}' AND e<='{right}')
              SELECT channel_id,{slot_sql('hh')} slot,item,
              sum(epoch(least(e,hh+INTERVAL '1 hour')-greatest(s,hh))/duration*{airing_weight}) n FROM hours GROUP BY 1,2,3'''
        else:
            assert anchor in ('start','midpoint')
            stamp='s' if anchor=='start' else "s+duration/2*INTERVAL '1 second'"
            query=f'''SELECT channel_id,
              {slot_sql(stamp)} slot,item,sum({airing_weight})
              FROM schedule WHERE item>0 AND s>='{left}' AND e<='{right}' GROUP BY 1,2,3'''
        for ch,slot,item,n in con.execute(query).fetchall():
            if item in model.ix:rr.append(ci[ch]*nslots+slot);cc.append(model.ix[item]);vv.append(n)
        return sparse.csr_matrix((vv,(rr,cc)),shape=(nc*nslots,len(model.items)))
    before=exposures(start,cutoff);after=exposures(cutoff,end)
    past=np.asarray(rate@before);future=np.asarray(rate@after)
    # Calibrate overall count scale using past qualified events, never future views.
    total=np.asarray(model.weights(half_life_days)[model.tx].sum(axis=1)).ravel()
    scale=np.divide(total,past.sum(axis=1),out=np.zeros_like(total),where=past.sum(axis=1)>0)
    return future*scale[:,None],past*scale[:,None]


def user_knn(model,neighbors=60):
    w=model.counts.copy();w.data=np.log1p(w.data)
    support=np.asarray((w>0).sum(axis=0)).ravel()
    # Count only users with pre-cutoff interactions; future-only empty rows must
    # not influence even a global IDF statistic.
    active_users=int(np.count_nonzero(np.diff(model.counts.indptr)))
    idf=np.log1p(active_users/(1+support))
    w=normalize(w.multiply(idf[None,:]),norm='l2').tocsr()
    sims=(w[model.tx]@w.T).toarray()
    rr,cc,vv=[],[],[]
    for i,own in enumerate(model.tx):
        sims[i,own]=0
        order=np.lexsort((np.arange(len(model.users)),-sims[i]))[:neighbors]
        for j in order:
            if sims[i,j]>0:rr.append(i);cc.append(j);vv.append(sims[i,j]**2)
    weights=normalize(sparse.csr_matrix((vv,(rr,cc)),shape=sims.shape),norm='l1')
    counts=normalize(model.counts,norm='l1')
    return (weights@counts[:,model.cx]).toarray()


def run():
    cfg=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v3_context');out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    users=sorted(set(r[0] for r in con.execute('SELECT DISTINCT user_id FROM logs').fetchall())|set(targets))
    results=[];predictions={}
    previous=json.loads(Path('runs/v2_experiments/results.json').read_text())
    for split in cfg['splits']:
        start,cutoff,end=cfg['train_start'],split['cutoff'],split['end']
        model=Models(train_rows(con,start,cutoff),load_metadata(con,cutoff),users,candidates_for(con,cutoff,end),dt.datetime.fromisoformat(cutoff),targets)
        truth,_=make_truth(con,cutoff,end,targets)
        history={u:int(model.history_n[model.ux[u]]) for u in targets}
        scores={};preds={}
        def add(name,score):
            t=time.perf_counter();pred=model.predict(score);metrics,_=evaluate(truth,pred,history)
            results.append({'split':split['name'],'method':name,**metrics,'ranking_seconds':time.perf_counter()-t})
            scores[name]=score;preds[name]=pred;dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        repeat=model.counts[model.tx].toarray()
        past_supply=item_supply(con,model,start,cutoff);future_supply=item_supply(con,model,cutoff,end)
        channel,_=channel_prior(con,model,start,cutoff,end)
        prior=channel/np.maximum(future_supply,1)[None,:]
        cosine=cosine_scores(model);add('item_cosine',cosine)
        add('user_knn',user_knn(model))
        for shrink in [20,50,100]:
            b=(repeat+shrink*prior)/(past_supply[None,:]+shrink)*future_supply[None,:]
            name=f'channel_bayes_{shrink}';add(name,b[:,model.cx])
            for k in [.1,.2,.3]:add(name+f'__itemcos_{k}',(1-k)*row_max_norm(scores[name])+k*row_max_norm(cosine))
            add(name+'__userknn_0.2',.8*row_max_norm(scores[name])+.2*row_max_norm(scores['user_knn']))
        for smooth in [0,7,21]:
            t=time.perf_counter();future,past=time_context(con,model,start,cutoff,end,smooth)
            print('context build seconds',round(time.perf_counter()-t,2),flush=True)
            name=f'time_{smooth}';add(name,future[:,model.cx])
            # Per-item lift above channel/time expected counts, regularized toward 1.
            for shrink in [1,5,20]:
                lifted=(repeat+shrink)/(past+shrink)*future
                add(name+f'__lift{shrink}',lifted[:,model.cx])
            time_prior=future/np.maximum(future_supply,1)[None,:]
            for shrink in [20,50]:
                b=(repeat+shrink*time_prior)/(past_supply[None,:]+shrink)*future_supply[None,:]
                n=name+f'__bayes{shrink}';add(n,b[:,model.cx])
                add(n+'__cos0.2',.8*row_max_norm(scores[n])+.2*row_max_norm(cosine))
            base=scores['channel_bayes_20__itemcos_0.2']
            add(name+'__v2_20',.8*row_max_norm(base)+.2*row_max_norm(scores[name]))
        predictions[split['name']]={'truth':truth,'predictions':preds}
        dump(out/'predictions.json',predictions)
        del scores,model
    combined=previous+results
    summary=[]
    for name in sorted({r['method'] for r in combined}):
        rows=[r for r in combined if r['method']==name]
        # The common item_cosine reference occurs in both runs; average per split once.
        split_scores={r['split']:r['map5'] for r in rows}
        summary.append({'method':name,'mean_map5':float(np.mean(list(split_scores.values()))),'split_scores':split_scores})
    summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
    print('TOP',json.dumps(summary[:15],indent=2),flush=True)


if __name__=='__main__':run()
