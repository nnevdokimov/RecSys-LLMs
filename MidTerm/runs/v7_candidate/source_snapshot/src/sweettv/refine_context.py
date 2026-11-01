"""Check airing-time integration and collaborative mixing before a new submission."""
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


def category_dominance(model):
    cats=sorted({m.get('tv_show_category','unknown') for m in model.metadata.values()})
    ix={c:j for j,c in enumerate(cats)}
    matrix=np.zeros((len(model.target_users),len(cats)))
    for j,item in enumerate(model.items):
        cat=model.metadata.get(item,{}).get('tv_show_category','unknown')
        if cat in ix:matrix[:,ix[cat]]+=model.counts[model.tx,j].toarray().ravel()
    total=matrix.sum(axis=1)
    dominant=np.divide(matrix.max(axis=1),total,out=np.zeros_like(total),where=total>0)
    return dominant,total


def run():
    cfg=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v4_refine');out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    users=sorted(set(r[0] for r in con.execute('SELECT DISTINCT user_id FROM logs').fetchall())|set(targets))
    results=[];predictions={}
    previous=json.loads(Path('runs/v3_context/summary.json').read_text())
    for split in cfg['splits']:
        start,cutoff,end=cfg['train_start'],split['cutoff'],split['end']
        model=Models(train_rows(con,start,cutoff),load_metadata(con,cutoff),users,candidates_for(con,cutoff,end),dt.datetime.fromisoformat(cutoff),targets)
        truth,_=make_truth(con,cutoff,end,targets);history={u:int(model.history_n[model.ux[u]]) for u in targets}
        repeat=model.counts[model.tx].toarray();cosine=cosine_scores(model)
        dominant,total=category_dominance(model)
        scores={};preds={}
        def add(name,score):
            pred=model.predict(score);metrics,_=evaluate(truth,pred,history)
            results.append({'split':split['name'],'method':name,**metrics});scores[name]=score;preds[name]=pred
            dump(out/'results.json',results);print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        for anchor in ['start','midpoint','mean']:
            future,past=time_context(con,model,start,cutoff,end,21,anchor)
            for shrink in [1,5]:
                lifted=(repeat+shrink)/(past+shrink)*future
                name=f'context21_{anchor}_lift{shrink}'
                add(name,lifted[:,model.cx])
                for weight in [.1,.2]:
                    add(name+f'__cos{weight}',(1-weight)*row_max_norm(scores[name])+weight*row_max_norm(cosine))
                adaptive=np.where((dominant>=.8)&(total>=50),.05,.2)[:,None]
                add(name+'__adaptive_cos',(1-adaptive)*row_max_norm(scores[name])+adaptive*row_max_norm(cosine))
            n1=f'context21_{anchor}_lift1';n5=f'context21_{anchor}_lift5'
            add(f'context21_{anchor}_ensemble',.5*row_max_norm(scores[n1])+.5*row_max_norm(scores[n5]))
        add('context21_three_anchors_ensemble',sum(row_max_norm(scores[f'context21_{anchor}_lift{shrink}']) for anchor in ['start','midpoint','mean'] for shrink in [1,5])/6)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
    summary=list(previous)
    for name in sorted({r['method'] for r in results}):
        rows=[r for r in results if r['method']==name]
        summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
    summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
    print('TOP',json.dumps(summary[:12],indent=2),flush=True)


if __name__=='__main__':run()
