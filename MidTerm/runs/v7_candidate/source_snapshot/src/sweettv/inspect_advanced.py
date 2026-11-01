"""Measure selected and standalone methods, retaining the same two example users."""
import datetime as dt
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from .advanced import als_scores,channel_prior,cosine_scores
from .context_experiment import time_context
from .experiment import candidates_for,evaluate,load_metadata,make_truth,train_rows
from .models import Models,row_max_norm
from .prepare import dump
from .refine_context import category_dominance


def run():
    out=Path('runs/v2_candidate');out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    start,cutoff,end='2020-03-09 00:00:00','2020-05-18 00:00:00','2020-07-27 00:00:00'
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    users=sorted(set(r[0] for r in con.execute('SELECT DISTINCT user_id FROM logs').fetchall())|set(targets))
    model=Models(train_rows(con,start,cutoff),load_metadata(con,cutoff),users,candidates_for(con,cutoff,end),dt.datetime.fromisoformat(cutoff),targets)
    truth,truth_n=make_truth(con,cutoff,end,targets);history={u:int(model.history_n[model.ux[u]]) for u in targets}
    scores={};preds={};aps={};measured=[]
    def add(name,fn):
        t=time.perf_counter();score=fn();pred=model.predict(score);metrics,ap=evaluate(truth,pred,history)
        elapsed=time.perf_counter()-t;scores[name]=score;preds[name]=pred;aps[name]=ap
        measured.append({'method':name,**metrics,'fit_and_rank_seconds':elapsed})
        print(name,metrics['map5'],elapsed,flush=True)
    add('popularity',model.popularity_scores)
    add('repeat',model.repeat)
    add('cbf28',lambda:model.content(28))
    add('ar20_28',lambda:model.association(20,half_life=28))
    add('als48',lambda:als_scores(model,factors=48,alpha=20,reg=5))
    add('item_cosine',lambda:cosine_scores(model))
    t=time.perf_counter();future,past=time_context(con,model,start,cutoff,end,21,'mean');context_seconds=time.perf_counter()-t
    add('time_context',lambda:future[:,model.cx])
    repeat=model.counts[model.tx].toarray();lifted=((repeat+1)/(past+1)*future)[:,model.cx]
    add('program_lift',lambda:lifted)
    dominant,total=category_dominance(model);weight=np.where((dominant>=.8)&(total>=50),.05,.2)[:,None]
    add('advanced_hybrid',lambda:(1-weight)*row_max_norm(lifted)+weight*row_max_norm(scores['item_cosine']))
    for result in measured:
        if result['method'] in ['time_context','program_lift']:result['fit_and_rank_seconds']+=context_seconds
        if result['method']=='advanced_hybrid':result['fit_and_rank_seconds']+=context_seconds+next(r['fit_and_rank_seconds'] for r in measured if r['method']=='item_cosine')
    old=json.loads(Path('runs/v4_refine/predictions.json').read_text())['original_10w']['predictions']['context21_mean_lift1__adaptive_cos']
    assert preds['advanced_hybrid']==old,'Validation-selected strategy did not reproduce'
    display=load_metadata(con,end)
    original_examples=json.loads(Path('artifacts/user_examples.json').read_text())
    examples=[]
    for example in original_examples:
        u=example['user_id'];j=targets.index(u)
        def desc(i):
            m=display.get(i,{})
            return {'id':int(i),'title':m.get('tv_show_title','нет метаданных'),'category':m.get('tv_show_category','нет категории'),'hit':i in truth[u]}
        examples.append({'user_id':u,'history_air_views':history[u],'history_unique_items':example['history_unique_items'],
            'history_top10':example['history_top10'],'truth':[{**desc(i),'air_views':n} for i,n in zip(truth[u],truth_n[u])],
            'dominant_category_share':float(dominant[j]),'cosine_weight':float(weight[j,0]),
            'recommendations':{name:[desc(i) for i in pred[u]] for name,pred in preds.items()},
            'ap5':{name:ap[u] for name,ap in aps.items()}})
    dump(out/'measured_methods.json',measured);dump(out/'user_examples.json',examples)
    dump(out/'validation_reproduction.json',{'selected_predictions_exactly_match_search':True,'context_build_seconds':context_seconds,'methods':len(measured)})


if __name__=='__main__':run()
