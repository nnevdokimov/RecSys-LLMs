"""Soft clock and recent-interest ablation; labels stay outside the fit boundary."""
import datetime as dt
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from .advanced import cosine_scores
from .context_experiment import time_context
from .experiment import candidates_for,evaluate,load_metadata,make_truth,train_rows
from .models import Models,row_max_norm
from .prepare import dump
from .refine_context import category_dominance


def run():
    protocol=json.loads(Path('configs/advanced.json').read_text())
    cfg=json.loads(Path('configs/soft_trend.json').read_text())
    out=Path('runs/v7_soft_trend');out.mkdir(parents=True,exist_ok=True)
    dump(out/'config_snapshot.json',cfg)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    previous=json.loads(Path('runs/v4_refine/predictions.json').read_text())
    results=[];all_preds={}
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end']
        users=sorted(set(r[0] for r in con.execute(f"SELECT DISTINCT user_id FROM logs WHERE s<'{cutoff}'").fetchall())|set(targets))
        model=Models(train_rows(con,start,cutoff),load_metadata(con,cutoff),users,candidates_for(con,cutoff,end),dt.datetime.fromisoformat(cutoff),targets)
        truth,_=make_truth(con,cutoff,end,targets)
        history={u:int(model.history_n[model.ux[u]]) for u in targets}
        repeat=model.counts[model.tx].toarray();cosine=cosine_scores(model)
        dominant,total=category_dominance(model)
        cw=np.where((dominant>=.8)&(total>=50),.05,.2)[:,None]
        preds={};contexts={}
        reference={'name':'v2_reference','bucket_hours':4,'day_mode':'weekend','shrink_days':21}
        for variant in [reference]+cfg['variants']:
            tick=time.perf_counter()
            future,past=time_context(con,model,start,cutoff,end,variant['shrink_days'],'mean',variant['bucket_hours'],variant['day_mode'],variant.get('smooth_sigma_hours',0),variant.get('half_life_days'))
            counts=model.weights(variant.get('half_life_days'))[model.tx].toarray()
            lifted=((counts+1)/(past+1)*future)[:,model.cx]
            contexts[variant['name']]=row_max_norm(lifted)
            score=(1-cw)*row_max_norm(lifted)+cw*row_max_norm(cosine)
            name=variant['name'];pred=model.predict(score)
            metrics,_=evaluate(truth,pred,history)
            results.append({'split':split['name'],'method':name,**metrics,'context_and_predict_seconds':time.perf_counter()-tick})
            preds[name]=pred;dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
            if name=='v2_reference':
                assert pred==previous[split['name']]['predictions']['context21_mean_lift1__adaptive_cos']
        blended=.5*contexts['v2_reference']+.5*contexts['soft3h_trend28']
        pred=model.predict((1-cw)*blended+cw*row_max_norm(cosine))
        metrics,_=evaluate(truth,pred,history)
        name='full_plus_soft_trend_50'
        results.append({'split':split['name'],'method':name,**metrics})
        preds[name]=pred;dump(out/'results.json',results)
        print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        all_preds[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',all_preds)
    summary=[]
    for name in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==name]
        summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
    summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
