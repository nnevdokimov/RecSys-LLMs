"""Supply-aware repeat, channel context, item cosine, and implicit ALS.

Writes experiments only to a new run directory. Never overwrites submitted v1.
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse
from scipy.linalg import solve
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .experiment import candidates_for, evaluate, load_metadata, make_truth, train_rows
from .models import Models, fusion, row_max_norm
from .prepare import dump, records


def item_supply(con, model, start, end):
    result=np.zeros(len(model.items))
    for i,n in con.execute(f"SELECT item,count(*) FROM schedule WHERE item>0 AND s>='{start}' AND e<='{end}' GROUP BY 1").fetchall():
        if i in model.ix:result[model.ix[i]]=n
    return result


def channel_prior(con,model,start,cutoff,end):
    """Smoothed per-airing rate by user/channel, future supply only ID/channel/time."""
    channels=sorted(x[0] for x in con.execute('SELECT DISTINCT channel_id FROM schedule').fetchall())
    ci={c:i for i,c in enumerate(channels)}
    supply=np.zeros(len(channels))
    for ch,n in con.execute(f"SELECT channel_id,count(*) FROM schedule WHERE item>0 AND s>='{start}' AND e<='{cutoff}' GROUP BY 1").fetchall():supply[ci[ch]]=n
    history=np.zeros((len(model.target_users),len(channels)))
    target_ix={u:i for i,u in enumerate(model.target_users)}
    for u,ch,n in con.execute(f"SELECT user_id,channel_id,count(*) FROM air_views WHERE fraction>=0.8 AND air_start>='{start}' AND air_end<='{cutoff}' GROUP BY 1,2").fetchall():
        if u in target_ix:history[target_ix[u],ci[ch]]=n
    rates=history/np.maximum(supply,1)[None,:]
    rr,cc,vv=[],[],[]
    for ch,i,n in con.execute(f"SELECT channel_id,item,count(*) FROM schedule WHERE item>0 AND s<'{end}' AND e>'{cutoff}' GROUP BY 1,2").fetchall():
        if i in model.ix:rr.append(ci[ch]);cc.append(model.ix[i]);vv.append(n)
    future=sparse.csr_matrix((vv,(rr,cc)),shape=(len(channels),len(model.items)))
    expected=np.asarray(rates@future)
    counts=np.asarray(future.sum(axis=0)).ravel()
    return expected,expected/np.maximum(counts,1)[None,:]


def cosine_scores(model,shrink=10,neighbors=80):
    b=(model.counts>0).astype(float)
    co=(b.T@b).tocsr();co.setdiag(0);co.eliminate_zeros()
    support=np.asarray(b.sum(axis=0)).ravel()
    for i in range(len(model.items)):
        a,z=co.indptr[i:i+2];idx=co.indices[a:z];count=co.data[a:z].copy()
        co.data[a:z]=count/np.sqrt(np.maximum(support[i]*support[idx],1))*count/(count+shrink)
        vals=co.data[a:z]
        if len(vals)>neighbors:
            order=np.lexsort((idx,-vals));mask=np.ones(len(vals),bool);mask[order[:neighbors]]=False;vals[mask]=0
    co.eliminate_zeros()
    w=model.counts[model.tx].copy();w.data=np.log1p(w.data)
    return (w@co[:,model.cx]).toarray()


def als_scores(model,factors=48,alpha=20,reg=5,iterations=12,seed=42,return_diagnostics=False):
    """Hu-Koren-Volinsky weighted least squares; confidence 1+alpha*log(1+n).

    Objective includes all unobserved user/item pairs with confidence 1.
    Each block minimization is an exact positive-definite linear solve.
    """
    counts=model.counts.tocsr()
    confidence=counts.copy();confidence.data=1+alpha*np.log1p(confidence.data)
    ct=confidence.T.tocsr()
    rng=np.random.default_rng(seed)
    x=rng.normal(0,.01,(counts.shape[0],factors))
    y=rng.normal(0,.01,(counts.shape[1],factors))
    eye=reg*np.eye(factors)
    def loss():
        total=float(np.sum((x.T@x)*(y.T@y)))
        for row in range(confidence.shape[0]):
            a,z=confidence.indptr[row:row+2];idx=confidence.indices[a:z];c=confidence.data[a:z]
            pred=y[idx]@x[row]
            total+=float(np.sum((c-1)*pred**2-2*c*pred+c))
        return total+reg*float(np.sum(x*x)+np.sum(y*y))
    def update(c,fixed,out):
        gram=fixed.T@fixed+eye
        for row in range(c.shape[0]):
            a,z=c.indptr[row:row+2]
            if a==z:out[row]=0;continue
            idx=c.indices[a:z];conf=c.data[a:z];f=fixed[idx]
            lhs=gram+(f.T*(conf-1))@f
            rhs=f.T@conf
            out[row]=solve(lhs,rhs,assume_a='pos',check_finite=False)
    with threadpool_limits(limits=1):
        losses=[loss()] if return_diagnostics else None
        for iteration in range(iterations):
            update(confidence,y,x);update(ct,x,y)
            if return_diagnostics:losses.append(loss())
        scores=x[model.tx]@y[model.cx].T
        return {'scores':scores,'objective_trace':losses} if return_diagnostics else scores


def run(config):
    cfg=json.loads(Path(config).read_text());out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True)
    con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    users=sorted(set(r[0] for r in con.execute('SELECT DISTINCT user_id FROM logs').fetchall())|set(targets))
    all_results=[];all_predictions={};run0=time.perf_counter()
    for split in cfg['splits']:
        cutoff,end=split['cutoff'],split['end'];start=cfg['train_start']
        truth,_=make_truth(con,cutoff,end,targets)
        pool=candidates_for(con,cutoff,end);meta=load_metadata(con,cutoff)
        train=train_rows(con,start,cutoff)
        model=Models(train,meta,users,pool,dt.datetime.fromisoformat(cutoff),targets)
        history={u:int(model.history_n[model.ux[u]]) for u in targets}
        predictions={};scores={};timings={}
        def add(name,fn):
            t=time.perf_counter();score=fn();pred=model.predict(score)
            metrics,ap=evaluate(truth,pred,history)
            elapsed=time.perf_counter()-t
            result={'split':split['name'],'method':name,**metrics,'seconds':elapsed,'eligible_users':sum(bool(x) for x in truth.values())}
            all_results.append(result);predictions[name]=pred;scores[name]=score;timings[name]=elapsed
            dump(out/'results.json',all_results)
            print(split['name'],name,f'MAP5={metrics["map5"]:.6f}',f'{elapsed:.2f}s',flush=True)
        add('popularity',model.popularity_scores)
        add('repeat',model.repeat)
        add('cbf28',lambda:model.content(28))
        add('ar20_28',lambda:model.association(20,half_life=28))
        add('v1_hybrid',lambda:fusion(scores['repeat'],scores['ar20_28'],scores['cbf28'],.6,.2,.2))
        supply_future=item_supply(con,model,cutoff,end)
        context,context_rate=channel_prior(con,model,start,cutoff,end)
        add('channel_expected',lambda:context[:,model.cx])
        for days in [None,42,56]:
            window_start=start if days is None else max(start,(dt.datetime.fromisoformat(cutoff)-dt.timedelta(days=days)).isoformat(sep=' '))
            tag='full' if days is None else str(days)
            if days is None:w=model.counts[model.tx].toarray()
            else:
                rows=train_rows(con,window_start,cutoff)
                rr,cc,vv=[],[],[]
                tix={u:i for i,u in enumerate(targets)}
                for r in rows:
                    if r['user_id'] in tix:rr.append(tix[r['user_id']]);cc.append(model.ix[r['item']]);vv.append(r['n'])
                w=sparse.csr_matrix((vv,(rr,cc)),shape=(len(targets),len(model.items))).toarray()
                add('repeat_'+tag,lambda:w[:,model.cx])
            past=item_supply(con,model,window_start,cutoff)
            ratio=(supply_future+1)/(past+1)
            for gamma in [.25,.5,1.0]:
                add(f'supply_{tag}_g{gamma}',lambda:w[:,model.cx]*ratio[model.cx][None,:]**gamma)
            for shrink in [5,20]:
                adjusted=(w+shrink*context_rate)/(past[None,:]+shrink)*supply_future[None,:]
                add(f'bayes_supply_{tag}_s{shrink}',lambda:adjusted[:,model.cx])
        add('item_cosine',lambda:cosine_scores(model))
        add('als48_a20_r5',lambda:als_scores(model,factors=48,alpha=20,reg=5,seed=cfg['seed']))
        add('als64_a40_r20',lambda:als_scores(model,factors=64,alpha=40,reg=20,seed=cfg['seed']))
        # Small predeclared combinations, all components evaluated alone above.
        for base in ['supply_full_g0.5','supply_full_g1.0','bayes_supply_full_s5','bayes_supply_full_s20','supply_42_g0.5']:
            for other,weight in [('item_cosine',.2),('als48_a20_r5',.2),('channel_expected',.2)]:
                add(base+'__'+other+'_20',lambda: .8*row_max_norm(scores[base])+.2*row_max_norm(np.maximum(scores[other],0)))
        for base in ['repeat','supply_full_g0.5','bayes_supply_full_s5']:
            add(base+'__als_cbf_60_20_20',lambda:.6*row_max_norm(scores[base])+.2*row_max_norm(np.maximum(scores['als48_a20_r5'],0))+.2*row_max_norm(scores['cbf28']))
        all_predictions[split['name']]={'truth':truth,'predictions':predictions}
        dump(out/'predictions.json',all_predictions)
        del scores,model
    names=sorted({x['method'] for x in all_results})
    summary=[]
    for name in names:
        rows=[r for r in all_results if r['method']==name]
        summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),
            'min_map5':float(min(r['map5'] for r in rows)),'split_scores':{r['split']:r['map5'] for r in rows},
            'seconds':sum(r['seconds'] for r in rows)})
    summary.sort(key=lambda x:-x['mean_map5'])
    dump(out/'summary.json',summary)
    dump(out/'run_info.json',{'config':cfg,'seconds':time.perf_counter()-run0,'v1_submitted_file_overwritten':False,
       'warning':'Validation periods overlap and historical labels were previously used; this is robustness checking, not an independent hidden test.'})
    print('TOP',json.dumps(summary[:10],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/advanced.json');run(p.parse_args().config)
