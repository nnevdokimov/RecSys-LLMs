"""Freeze the validation-selected advanced strategy and produce its exact CSV."""
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import duckdb
import numpy as np

from .advanced import cosine_scores
from .context_experiment import time_context
from .experiment import candidates_for,load_metadata,train_rows,write_submission
from .metrics import ap_at_k
from .models import Models,row_max_norm
from .prepare import dump
from .refine_context import category_dominance


def final_scores(con,model,start,cutoff,end):
    future,past=time_context(con,model,start,cutoff,end,21,'mean')
    repeat=model.counts[model.tx].toarray()
    context=((repeat+1)/(past+1)*future)[:,model.cx]
    cosine=cosine_scores(model)
    dominance,total=category_dominance(model)
    weight=np.where((dominance>=.8)&(total>=50),.05,.2)[:,None]
    return (1-weight)*row_max_norm(context)+weight*row_max_norm(cosine)


def run():
    out=Path('runs/v2_candidate');out.mkdir(parents=True,exist_ok=True)
    summary=json.loads(Path('runs/v4_refine/summary.json').read_text())
    selected=summary[0]
    expected='context21_mean_lift1__adaptive_cos'
    assert selected['method']==expected,selected
    # Selection is frozen and saved before any prediction for weeks 31-42.
    old=json.loads(Path('runs/v2_experiments/predictions.json').read_text())
    new=json.loads(Path('runs/v4_refine/predictions.json').read_text())
    differences={};rng=np.random.default_rng(42)
    for split,data in new.items():
        eligible=[u for u,r in data['truth'].items() if r]
        delta=np.array([ap_at_k(data['truth'][u],data['predictions'][expected][u])-ap_at_k(data['truth'][u],old[split]['predictions']['v1_hybrid'][u]) for u in eligible])
        boot=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(2000)])
        differences[split]={'users':len(eligible),'mean_delta_vs_v1':float(delta.mean()),'descriptive_paired_95_interval':np.quantile(boot,[.025,.975]).tolist()}
    selection={'method':expected,'mean_rolling_map5':selected['mean_map5'],'split_scores':selected['split_scores'],
       'criterion':'Highest mean MAP@5 over the three predefined rolling splits',
       'private_score_used_for_selection':False,'candidate_private_score':None,
       'comparisons_vs_v1':differences,
       'caveat':'Intervals computed after extensive model selection on overlapping historical windows. Not an independent significance test.',
       'parameters':{'context_training_window':'all allowed past','context_features':['channel','weekday/weekend','4-hour bucket'],
          'raw_view_context_shrink_days':21,'airing_time_integration':'duration-weighted mean of hourly contexts',
          'program_lift':'(qualified_view_count+1)/(past_context_expected_count+1)',
          'item_similarity':'binary co-view cosine multiplied by co_support/(co_support+10), top 80 neighbors; no diagonal',
          'history_similarity_weight':'log(1+qualified_view_count)',
          'cosine_component_weight':{'default':.2,'dominant_category_share>=0.8_and_qualified_air_views>=50':.05},
          'fusion':'per-user max-normalized context score and item-cosine score; shared popularity fallback'},
       'selection_configurations_checked':len({r['method'] for r in summary})}
    dump(out/'selection.json',selection)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    start,cutoff,end='2020-03-09 00:00:00','2020-07-27 00:00:00','2020-10-19 00:00:00'
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    users=sorted(set(r[0] for r in con.execute('SELECT DISTINCT user_id FROM logs').fetchall())|set(targets))
    t=time.perf_counter()
    model=Models(train_rows(con,start,cutoff),load_metadata(con,cutoff),users,candidates_for(con,cutoff,end),dt.datetime.fromisoformat(cutoff),targets)
    pred=model.predict(final_scores(con,model,start,cutoff,end))
    candidate=out/'submission_v2.csv';tmp=Path('tmp/submission_v2_reproduced.csv')
    checks=write_submission('sweettv-tv-program-recommender/submission.csv',pred,tmp,set(model.candidates))
    if candidate.exists():assert candidate.read_bytes()==tmp.read_bytes(),'Frozen candidate changed'
    else:candidate.write_bytes(tmp.read_bytes())
    original_test={int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM schedule_raw WHERE filename LIKE '%31-42%' AND tv_show_id!='0'").fetchall()}
    assert all(i in original_test for ids in pred.values() for i in ids)
    checks.update({'path':str(candidate),'all_ids_in_original_test_schedule':True,'train_start':start,'cutoff':cutoff,'test_end':end,
       'fit_and_predict_seconds':time.perf_counter()-t,'model':expected,'v1_sha256_unchanged':hashlib.sha256(Path('artifacts/final_submission.csv').read_bytes()).hexdigest()=='f6f31a19e73b0b6f93db5cc63f116a6136cb515763552c97948462c979086ea5'})
    dump(out/'submission_checks.json',checks)
    print(json.dumps({'selection':selection,'submission':checks},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
