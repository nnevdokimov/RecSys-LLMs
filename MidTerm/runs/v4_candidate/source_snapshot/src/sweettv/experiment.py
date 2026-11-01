"""One temporal hold-out, predeclared small grid, frozen selection, final refit."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import resource
import time
from pathlib import Path

import duckdb
import numpy as np

from .metrics import ap_at_k
from .models import Models, fusion
from .prepare import dump, records


def load_metadata(con, cutoff):
    # Each field's modal value is estimated solely from pre-cutoff airings.
    fields=['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director','actors']
    result={}
    for field in fields:
        rows=con.execute(f'''SELECT item,{field} FROM (
           SELECT item,{field},row_number() OVER(PARTITION BY item ORDER BY count(*) DESC,{field}) rn
           FROM schedule WHERE item>0 AND s<'{cutoff}' AND {field} IS NOT NULL GROUP BY 1,2) WHERE rn=1''').fetchall()
        for item,value in rows:
            result.setdefault(item,{})[field]=value
    return result


def train_rows(con,start,cutoff,threshold='fraction>=0.8'):
    return records(con,f'''SELECT user_id,item,count(*) n,sum(watched_seconds) watched_total_seconds,max(air_end) last_time,
      sum(pow(2.0,-epoch(TIMESTAMP '{cutoff}'-air_end)/86400/28)) n28
      FROM air_views WHERE air_start>='{start}' AND air_end<='{cutoff}' AND {threshold} GROUP BY 1,2''')


def make_truth(con,start,end,targets):
    rows=con.execute(f'''SELECT user_id,item,n FROM (
      SELECT user_id,item,count(*) n,row_number() OVER(PARTITION BY user_id ORDER BY count(*) DESC,item) rn
      FROM air_views WHERE air_start>='{start}' AND air_end<='{end}' AND fraction>=0.8 GROUP BY 1,2) WHERE rn<=5 ORDER BY user_id,rn''').fetchall()
    truth={u:[] for u in targets}
    counts={u:[] for u in targets}
    for u,i,n in rows:
        if u in truth: truth[u].append(i);counts[u].append(n)
    return truth,counts


def candidates_for(con,start,end):
    return {r[0] for r in con.execute(f"SELECT DISTINCT item FROM schedule WHERE item>0 AND s<'{end}' AND e>'{start}'").fetchall()}


def evaluate(truth,predictions,history_sizes):
    eligible=[u for u in truth if truth[u]]
    all_ap={u:ap_at_k(truth[u],predictions[u]) for u in truth}
    cold=[u for u in eligible if history_sizes[u]==0]
    warm=[u for u in eligible if history_sizes[u]>0]
    five=[u for u in eligible if len(truth[u])==5]
    return {'map5':float(np.mean([all_ap[u] for u in eligible])),
            'map5_all_users_empty_zero':float(np.mean(list(all_ap.values()))),
            'map5_gt5':float(np.mean([all_ap[u] for u in five])) if five else None,
            'map5_cold':float(np.mean([all_ap[u] for u in cold])) if cold else None,
            'map5_warm':float(np.mean([all_ap[u] for u in warm])) if warm else None},all_ap


def fit_grid(model):
    outputs={};scores={};times={}
    funcs=[('popularity',model.popularity_scores),('repeat',model.repeat),('repeat28',lambda:model.repeat(28)),
       ('content',model.content),('content28',lambda:model.content(28)),
       ('association5',lambda:model.association(5)),('association20',lambda:model.association(20)),
       ('association20_28',lambda:model.association(20,half_life=28))]
    for name,fn in funcs:
        t=time.perf_counter();scores[name]=fn();outputs[name]=model.predict(scores[name]);times[name]=time.perf_counter()-t
        print('Fitted',name,round(times[name],3),flush=True)
    for name,r,a,c in [('hybrid80_20',.8,.2,0),('hybrid90_10',.9,.1,0),('hybrid80_10_10',.8,.1,.1),('hybrid60_20_20',.6,.2,.2)]:
        t=time.perf_counter()
        s=fusion(scores['repeat28'],scores['association20_28'],scores['content28'],r,a,c)
        outputs[name]=model.predict(s)
        times[name]=time.perf_counter()-t+times['repeat28']+times['association20_28']+(times['content28'] if c else 0)
    # Recorded refinement: repeat28 underperformed repeat; vary only this component.
    for name,r,a,c in [('hybrid80_10_10_static_repeat',.8,.1,.1),('hybrid60_20_20_static_repeat',.6,.2,.2)]:
        t=time.perf_counter()
        s=fusion(scores['repeat'],scores['association20_28'],scores['content28'],r,a,c)
        outputs[name]=model.predict(s)
        times[name]=time.perf_counter()-t+times['repeat']+times['association20_28']+times['content28']
    return outputs,times


def selected_scores(m,name):
    if name=='popularity': return m.popularity_scores()
    if name in ('repeat','repeat28'): return m.repeat(28 if name.endswith('28') else None)
    if name in ('content','content28'): return m.content(28 if name.endswith('28') else None)
    if name.startswith('association'):
        return m.association(5 if name=='association5' else 20,half_life=28 if name.endswith('_28') else None)
    base_name=name.removesuffix('_static_repeat')
    weights={'hybrid80_20':(.8,.2,0),'hybrid90_10':(.9,.1,0),'hybrid80_10_10':(.8,.1,.1),'hybrid60_20_20':(.6,.2,.2)}[base_name]
    r=m.repeat(None if name.endswith('_static_repeat') else 28);a=m.association(20,half_life=28)
    c=m.content(28) if weights[2] else np.zeros_like(r)
    return fusion(r,a,c,*weights)


def write_submission(template,predictions,path,candidates):
    with Path(template).open(newline='') as f:
        source=list(csv.DictReader(f))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['user_id','tv_show_id'],lineterminator='\n');w.writeheader()
        for row in source:
            ids=predictions[row['user_id']]
            assert len(ids)==5 and len(set(ids))==5 and all(isinstance(i,int) and i>0 and i in candidates for i in ids)
            w.writerow({'user_id':row['user_id'],'tv_show_id':' '.join(map(str,ids))})
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;out=list(reader)
    assert columns==['user_id','tv_show_id'] and len(out)==len(source)
    assert [r['user_id'] for r in out]==[r['user_id'] for r in source]
    assert len({r['user_id'] for r in out})==len(source)
    for row in out:
        parts=row['tv_show_id'].split(' ')
        assert len(parts)==5 and all(p.isdigit() and int(p)>0 and int(p) in candidates for p in parts)
    return {'path':str(path),'rows':len(out),'columns':columns,'five_unique_positive_candidate_ids':True,'template_user_order_preserved':True,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def run(config):
    t0=time.perf_counter();cfg=json.loads(Path(config).read_text())
    np.random.seed(cfg['seed'])
    con=duckdb.connect('artifacts/data.duckdb',read_only=True)
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    users=sorted(set(r[0] for r in con.execute('SELECT DISTINCT user_id FROM logs').fetchall())|set(targets))
    start,cutoff,end=cfg['train_start'],cfg['validation_start'],cfg['validation_end']
    rows=train_rows(con,start,cutoff)
    meta=load_metadata(con,cutoff)
    pool=candidates_for(con,cutoff,end)
    truth,truth_counts=make_truth(con,cutoff,end,targets)
    model=Models(rows,meta,users,pool,dt.datetime.fromisoformat(cutoff),targets)
    history={u:int(model.history_n[model.ux[u]]) for u in targets}
    outputs,times=fit_grid(model)
    results=[];aps={}
    for name,preds in outputs.items():
        metrics,aps[name]=evaluate(truth,preds,history)
        results.append({'method':name,**metrics,'fit_and_predict_seconds':times[name]})
    # One-variable sensitivity: train with absolute 300 s, same truth/users/pool.
    absolute_rows=train_rows(con,start,cutoff,'watched_seconds>=300')
    am=Models(absolute_rows,meta,users,pool,dt.datetime.fromisoformat(cutoff),targets)
    t=time.perf_counter();apreds=am.predict(am.repeat(28));at=time.perf_counter()-t
    metrics,aps['repeat28_absolute300']=evaluate(truth,apreds,history)
    outputs['repeat28_absolute300']=apreds
    results.append({'method':'repeat28_absolute300',**metrics,'fit_and_predict_seconds':at})
    # Sensitivity has different training labels; it is diagnostic, not in the main selection grid.
    eligible_results=[r for r in results if r['method']!='repeat28_absolute300']
    winner=max(eligible_results,key=lambda r:(r['map5'],-r['fit_and_predict_seconds']))['method']
    eligible=[u for u in targets if truth[u]]
    cold=[u for u in eligible if history[u]==0]
    ground_items=[i for u in eligible for i in truth[u]]
    protocol={'train_start':start,'cutoff':cutoff,'validation_end':end,'target_users':len(targets),'training_users_total':len(users),
       'eligible_users':len(eligible),'users_without_ground_truth':len(targets)-len(eligible),'users_with_5_ground_truth':sum(len(truth[u])==5 for u in targets),
       'candidate_count':len(pool),'candidate_gt_coverage':sum(i in pool for i in ground_items)/len(ground_items),
       'train_user_item_pairs':len(rows),'train_air_views':sum(r['n'] for r in rows),
       'target_users_without_training_history':sum(history[u]==0 for u in targets),'eligible_cold_users':len(cold),
       'candidate_items_without_training_metadata':len(pool-set(meta)),
       'ground_truth_seen_in_train_fraction':sum(i in set(model.items[j] for j in model.counts[model.ux[u]].indices) for u in eligible for i in truth[u])/len(ground_items),
       'primary_metric':'MAP@5 over template users with >=1 labelled relevant item; empty users reported separately as zero',
       'truth_rule':'Top 5 unique IDs by number of airings with union watch fraction >= 0.8, ties item ID ascending',
       'allow_seen':True,'schedule_policy':'Future schedule supplies candidate IDs and airing intervals only; all scoring metadata and IDF are pre-cutoff',
       'boundary_rule':'Only airings wholly within a train or validation interval can supply a label',
       'boundary_crossing_user_air_pairs':con.execute(f"SELECT count(*) FROM air_views WHERE air_start<'{cutoff}' AND air_end>'{cutoff}'").fetchone()[0]}
    # Paired bootstrap is descriptive after grid selection, not an unbiased test.
    rng=np.random.default_rng(cfg['seed'])
    delta=np.array([aps[winner][u]-aps['popularity'][u] for u in eligible])
    boot=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(2000)])
    comparison={'selected':winner,'selection_metric':'primary hold-out MAP@5','selection_uses_private_score':False,
        'delta_vs_popularity':float(delta.mean()),'paired_bootstrap_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),
        'interval_caveat':'Descriptive same hold-out interval after model selection, not corrected for selection/multiple comparisons.'}
    delta_repeat=np.array([aps[winner][u]-aps['repeat'][u] for u in eligible])
    boot_repeat=np.array([rng.choice(delta_repeat,size=len(delta_repeat),replace=True).mean() for _ in range(2000)])
    comparison['delta_vs_repeat']=float(delta_repeat.mean())
    comparison['paired_interval_vs_repeat']=np.quantile(boot_repeat,[.025,.975]).tolist()
    # Fixed examples: heaviest target history, and shortest nonempty target history with truth.
    heavy=max(eligible,key=lambda u:(history[u],u))
    short=min([u for u in eligible if 0<history[u] and u!=heavy],key=lambda u:(history[u],u))
    examples=[]
    display_meta=load_metadata(con,end)  # Display only; never passed to validation models.
    for u in [heavy,short]:
        train=sorted([r for r in rows if r['user_id']==u],key=lambda r:(-r['n'],r['item']))
        def describe(i):
            d=display_meta.get(i,{})
            return {'id':int(i),'title':d.get('tv_show_title','нет метаданных'),'category':d.get('tv_show_category','нет категории')}
        examples.append({'user_id':u,'history_air_views':history[u],'history_unique_items':len(train),
          'history_top10':[{**describe(r['item']),'air_views':r['n']} for r in train[:10]],
          'truth':[{**describe(i),'air_views':n} for i,n in zip(truth[u],truth_counts[u])],
          'recommendations':{name:[{**describe(i),'hit':i in truth[u]} for i in outputs[name][u]] for name in outputs},
          'ap5':{name:aps[name][u] for name in outputs}})
    dump('artifacts/validation_protocol.json',protocol);dump('artifacts/experiments.json',results)
    dump('artifacts/selection.json',comparison);dump('artifacts/user_examples.json',examples)
    dump('artifacts/validation_predictions.json',{'truth':truth,'predictions':outputs,'ap5':aps})
    print(json.dumps({'protocol':protocol,'results':results,'selection':comparison},ensure_ascii=False,indent=2),flush=True)
    # Freeze selected method before any test predictions or submission scoring.
    all_rows=train_rows(con,start,end)
    test_pool=candidates_for(con,end,cfg['test_end'])
    final_meta=load_metadata(con,end)
    fm=Models(all_rows,final_meta,users,test_pool,dt.datetime.fromisoformat(end),targets)
    t=time.perf_counter();final_predictions=fm.predict(selected_scores(fm,winner));final_seconds=time.perf_counter()-t
    out=Path('artifacts/final_submission.csv')
    checks=write_submission(Path(cfg['data_dir'])/'submission.csv',final_predictions,out,test_pool)
    checks.update({'method':winner,'candidate_count':len(test_pool),'fit_and_predict_seconds':final_seconds,
       'training_air_views':sum(r['n'] for r in all_rows),'training_user_item_pairs':len(all_rows),
       'cold_target_users':int(sum(fm.history_n[fm.ux[u]]==0 for u in targets)),
       'candidate_items_without_training_metadata':len(test_pool-set(final_meta)),
       'total_experiment_seconds':time.perf_counter()-t0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    dump('artifacts/submission_checks.json',checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
    con.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment.json');run(p.parse_args().config)
