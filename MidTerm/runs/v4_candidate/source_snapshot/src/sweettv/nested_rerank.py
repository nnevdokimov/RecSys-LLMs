"""Train a pointwise hybrid reranker on earlier windows inside each train split.

No outer hold-out label is passed to fit(). Cached features are computed before
loading their labels, and their cache key includes the complete implementation.
"""
import datetime as dt
import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .advanced import cosine_scores,item_supply
from .context_experiment import time_context
from .category_context import category_time_features
from .experiment import candidates_for,evaluate,load_metadata,make_truth,train_rows
from .models import Models,row_max_norm,top_five
from .prepare import dump
from .refine_context import category_dominance


def inner_windows(start,cutoff,cfg):
    left=dt.datetime.fromisoformat(start);right=dt.datetime.fromisoformat(cutoff)
    anchor=left+dt.timedelta(days=cfg['inner_min_train_days']);result=[]
    while (right-anchor).days>=cfg['inner_min_forecast_days']:
        end=min(right,anchor+dt.timedelta(days=cfg['inner_max_forecast_days']))
        result.append((str(anchor),str(end)))
        anchor+=dt.timedelta(days=cfg['inner_step_days'])
    assert all(left<dt.datetime.fromisoformat(a)<dt.datetime.fromisoformat(b)<=right for a,b in result)
    return result


def build_features(con,start,cutoff,end,targets,cfg):
    # Cache is an acceleration only; all contributors are hashed into the key.
    fingerprint=hashlib.sha256()
    for path in sorted(Path('src/sweettv').glob('*.py')):fingerprint.update(path.read_bytes())
    fingerprint.update(json.dumps([start,cutoff,end,targets,cfg],sort_keys=True).encode())
    folder=Path('tmp/nested_features');folder.mkdir(parents=True,exist_ok=True)
    path=folder/(fingerprint.hexdigest()+'.npz')
    if path.exists():
        with np.load(path) as data:return {k:data[k] for k in data.files}
    tick=time.perf_counter()
    users=sorted(set(r[0] for r in con.execute(f"SELECT DISTINCT user_id FROM logs WHERE s<'{cutoff}'").fetchall())|set(targets))
    rows=train_rows(con,start,cutoff)
    m=Models(rows,load_metadata(con,cutoff),users,candidates_for(con,cutoff,end),dt.datetime.fromisoformat(cutoff),targets)
    repeat=m.repeat();recent=m.repeat(28);cosine=cosine_scores(m)
    future,past=time_context(con,m,start,cutoff,end,21,'mean')
    lifted=((m.counts[m.tx].toarray()+1)/(past+1)*future)[:,m.cx]
    soft_future,soft_past=time_context(con,m,start,cutoff,end,21,'mean',1,'weekend',3,28)
    trend_lift=((m.weights(28)[m.tx].toarray()+1)/(soft_past+1)*soft_future)[:,m.cx]
    dominance,total=category_dominance(m);cw=np.where((dominance>=.8)&(total>=50),.05,.2)[:,None]
    baseline=(1-cw)*row_max_norm(lifted)+cw*row_max_norm(cosine)
    components={'v2':baseline,'raw_time':future[:,m.cx],'repeat':repeat,'item_cosine':cosine,
                'content':m.content(28),'association':m.association(20,half_life=28),'popularity':m.popularity_scores()}
    selected=[]
    for i in range(len(targets)):
        chosen=set()
        for name,k in cfg['candidate_top_k'].items():
            order=np.lexsort((m.candidates,-m.pop,-components[name][i]))[:k]
            chosen.update(order.tolist())
        selected.extend((i,j) for j in sorted(chosen))
    rr,cc=np.asarray(selected,dtype=np.int32).T
    values=[];names=[]
    def add(name,array):
        values.append(np.broadcast_to(array,baseline.shape)[rr,cc]);names.append(name)
    for name,array in {**components,'repeat28':recent,'lifted':lifted,'trend_soft':trend_lift}.items():
        add('log_'+name,np.log1p(array));add('relative_'+name,row_max_norm(array))
        order=np.argsort(-array,axis=1,kind='stable')
        ranks=np.empty_like(order);np.put_along_axis(ranks,order,np.arange(array.shape[1])[None,:]+1,axis=1)
        add('reciprocal_rank_'+name,1/ranks)
    add('log_expected_past',np.log1p(past[:,m.cx]))
    add('log_supply_before',np.log1p(item_supply(con,m,start,cutoff)[m.cx])[None,:])
    add('log_supply_after',np.log1p(item_supply(con,m,cutoff,end)[m.cx])[None,:])
    age=np.zeros_like(repeat)
    target_ix={u:i for i,u in enumerate(targets)};candidate_ix={int(item):j for j,item in enumerate(m.candidates)}
    cut=dt.datetime.fromisoformat(cutoff)
    for row in rows:
        if row['user_id'] in target_ix and row['item'] in candidate_ix:
            age[target_ix[row['user_id']],candidate_ix[row['item']]]=2**(-max(0,(cut-row['last_time']).total_seconds()/86400)/28)
    add('last_event_recency',age)
    add('user_log_events',np.log1p(total)[:,None])
    add('user_log_distinct',np.log1p(m.history_items[m.tx])[:,None])
    add('category_dominance',dominance[:,None])
    add('train_days',np.array((cut-dt.datetime.fromisoformat(start)).days))
    add('forecast_days',np.array((dt.datetime.fromisoformat(end)-cut).days))
    if cfg.get('category_context'):
        for name,array in category_time_features(con,m,start,cutoff,end,**cfg['category_context']).items():
            add(name,array)
    result={'x':np.column_stack(values).astype(np.float32),'rr':rr,'cc':cc,
            'base':baseline[rr,cc].astype(np.float32),'items':m.candidates,'pop':m.pop,
            'history':m.history_n[m.tx],'feature_names':np.array(names)}
    assert np.isfinite(result['x']).all()
    np.savez_compressed(path,**result)
    print('FEATURES',cutoff,end,len(rr),'seconds',round(time.perf_counter()-tick,2),flush=True)
    return result


def label_rows(data,truth,targets):
    relevant=[set(truth[u]) for u in targets]
    label=np.fromiter((int(int(data['items'][c]) in relevant[r]) for r,c in zip(data['rr'],data['cc'])),dtype=np.int8)
    valid=np.array([bool(relevant[r]) and data['history'][r]>0 for r in data['rr']])
    covered=sum(len(relevant[i]&set(data['items'][data['cc'][data['rr']==i]].tolist())) for i in range(len(targets)))
    denominator=sum(map(len,relevant))
    return label,valid,covered/max(1,denominator)


def predict(data,targets,reranker,weight):
    prob=reranker.predict_proba(data['x'])[:,1]
    score=np.zeros((len(targets),len(data['items'])))
    learned=np.zeros_like(score);learned[data['rr'],data['cc']]=prob
    base=np.zeros_like(score);base[data['rr'],data['cc']]=data['base']
    score=weight*row_max_norm(learned)+(1-weight)*row_max_norm(base)
    score[data['history']==0]=0 # same popularity fallback for no-history users
    return {u:top_five(score[i],data['items'],data['pop']) for i,u in enumerate(targets)}


def fit_before(con,start,cutoff,targets,cfg):
    xx=[];yy=[];audit=[]
    for anchor,inner_end in inner_windows(start,cutoff,cfg):
        data=build_features(con,start,anchor,inner_end,targets,cfg)
        truth,_=make_truth(con,anchor,inner_end,targets)
        y,valid,coverage=label_rows(data,truth,targets);xx.append(data['x'][valid]);yy.append(y[valid])
        audit.append({'feature_cutoff':anchor,'label_end':inner_end,'rows':int(valid.sum()),'positive_rows':int(y[valid].sum()),'candidate_gt_coverage':coverage})
    tick=time.perf_counter();clf=HistGradientBoostingClassifier(**cfg['estimator'],random_state=cfg['seed'])
    clf.fit(np.concatenate(xx),np.concatenate(yy))
    return clf,{'outer_cutoff':cutoff,'inner_windows':audit,'fit_seconds':time.perf_counter()-tick,'outer_labels_used_in_fit':False}


def run(config_path='configs/nested_rerank.json',out_path='runs/v8_nested'):
    cfg=json.loads(Path(config_path).read_text());protocol=json.loads(Path('configs/advanced.json').read_text())
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True);dump(out/'config_snapshot.json',cfg)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    results=[];predictions={};fit_audit={}
    with threadpool_limits(limits=4):
        for split in protocol['splits']:
            start,cutoff,end=protocol['train_start'],split['cutoff'],split['end']
            clf,fit_audit[split['name']]=fit_before(con,start,cutoff,targets,cfg)
            fit_seconds=fit_audit[split['name']]['fit_seconds']
            dump(out/'fit_audit.json',fit_audit)
            data=build_features(con,start,cutoff,end,targets,cfg)
            # The outer hold-out labels are read only after fitting is complete.
            truth,_=make_truth(con,cutoff,end,targets);preds={}
            _,_,coverage=label_rows(data,truth,targets)
            for weight in cfg['blends']:
                name=f'nested_hgb_weight{weight}';pred=predict(data,targets,clf,weight)
                metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())))
                results.append({'split':split['name'],'method':name,**metrics,'candidate_gt_coverage':coverage,'fit_seconds':fit_seconds})
                preds[name]=pred;dump(out/'results.json',results)
                print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
            predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
            dump(out/'feature_names.json',data['feature_names'].tolist())
    summary=[]
    for name in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==name]
        summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
    summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/nested_rerank.json');parser.add_argument('--out',default='runs/v8_nested')
    args=parser.parse_args();run(args.config,args.out)
