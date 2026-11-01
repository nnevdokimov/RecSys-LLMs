"""Small predeclared drift check: full history versus recent clock/interest profiles."""
import datetime as dt
import json
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
    protocol=json.loads(Path('configs/advanced.json').read_text());cfg=json.loads(Path('configs/drift.json').read_text())
    out=Path('runs/v5_drift');out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    previous=json.loads(Path('runs/v4_refine/predictions.json').read_text())
    results=[];all_preds={}
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end']
        users=sorted(set(r[0] for r in con.execute(f"SELECT DISTINCT user_id FROM logs WHERE s<'{cutoff}'").fetchall())|set(targets))
        pool=candidates_for(con,cutoff,end);metadata=load_metadata(con,cutoff)
        model=Models(train_rows(con,start,cutoff),metadata,users,pool,dt.datetime.fromisoformat(cutoff),targets)
        truth,_=make_truth(con,cutoff,end,targets);history={u:int(model.history_n[model.ux[u]]) for u in targets}
        cosine=cosine_scores(model);dominant,total=category_dominance(model)
        cw=np.where((dominant>=.8)&(total>=50),.05,.2)[:,None]
        future,past=time_context(con,model,start,cutoff,end,21,'mean')
        full=((model.counts[model.tx].toarray()+1)/(past+1)*future)[:,model.cx]
        preds={}
        def add(name,context):
            score=(1-cw)*row_max_norm(context)+cw*row_max_norm(cosine)
            pred=model.predict(score);metrics,_=evaluate(truth,pred,history)
            results.append({'split':split['name'],'method':name,**metrics});preds[name]=pred
            dump(out/'results.json',results);print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        add('v2_reference',full)
        assert preds['v2_reference']==previous[split['name']]['predictions']['context21_mean_lift1__adaptive_cos']
        for days in cfg['windows_days']:
            recent_start=max(start,(dt.datetime.fromisoformat(cutoff)-dt.timedelta(days=days)).isoformat(sep=' '))
            recent=Models(train_rows(con,recent_start,cutoff),metadata,users,pool,dt.datetime.fromisoformat(cutoff),targets)
            rf,rp=time_context(con,recent,recent_start,cutoff,end,21,'mean')
            rc=((recent.counts[recent.tx].toarray()+1)/(rp+1)*rf)[:,recent.cx]
            for weight in cfg['full_recent_mix_weights']:
                add(f'recent{days}_mix{weight}',(1-weight)*row_max_norm(full)+weight*row_max_norm(rc))
        all_preds[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',all_preds)
    summary=[]
    for name in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==name]
        summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
    summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
