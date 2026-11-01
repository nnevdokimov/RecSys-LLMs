"""Freeze final comparison winner, audit a reserved cohort, reproduce exact CSV."""
import hashlib,json,time
from pathlib import Path
import duckdb
import numpy as np
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth,write_submission
from sweettv.metrics import ap_at_k
from sweettv.prepare import dump
from sweettv.models import top_five
from final_ranker import feature_data,fit_ranker,predict_ranker,predict_reference

out=Path('runs/v5_candidate');out.mkdir(exist_ok=True)
cfg=json.loads(Path('configs/rank_final_check.json').read_text());rank_cfg=json.loads(Path(cfg['ranker_config']).read_text());feature_cfg=json.loads(Path(cfg['feature_config']).read_text())
selected=json.loads(Path('runs/v19_rank_final/summary.json').read_text())[0]
assert selected['mean_map5']>=0.42011941589098994-1e-12
variant=next(v for v in cfg['variants'] if v['name']==selected['method'])
params={**rank_cfg['common'],'n_estimators':150,**{k:v for k,v in variant.items() if k!='name'}}
if params['objective']=='rank_xendcg':params.pop('lambdarank_truncation_level')
vocabulary=json.loads(Path('runs/v19_rank_final/vocabulary.json').read_text())
selection={**selected,'source_folder':'runs/v19_rank_final','parameters':params,'feature_config':cfg['feature_config'],'feature_count':86,'vocabulary':vocabulary,
 'criterion':'Highest mean across predefined three historical rolling splits; broad search and overlapping windows disclosed',
 'reserve_user_labels_used_for_choice':False,'private_score_used_for_parameter_selection':False,'private_score':None}
p=out/'selection.json'
if p.exists():assert json.loads(p.read_text())==selection
else:dump(p,selection) # before reserve labels and before final test predictions
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
start='2020-03-09 00:00:00';cutoff='2020-05-18 00:00:00';end='2020-07-27 00:00:00'
with threadpool_limits(limits=4):
    ranker,classifier,audit=fit_ranker(con,start,cutoff,targets,feature_cfg,vocabulary,params,True)
    data=feature_data(con,start,cutoff,end,targets,feature_cfg,vocabulary)
    new=predict_ranker(data,targets,ranker);old=predict_reference(data,targets,classifier)
    expected=json.loads(Path('runs/v19_rank_final/predictions.json').read_text())['original_10w']['predictions'][selected['method']]
    expected_old=json.loads(Path('runs/v10_category/predictions.json').read_text())['original_10w']['predictions']['nested_hgb_weight1.0']
    assert new==expected and old==expected_old
    dump(out/'validation_reproduction.json',{'new_all_1260_lists_identical':True,'reference_all_1260_lists_identical':True,'fit_audit':audit})
    if not (out/'reserve_evaluation.json').exists():
        # This cohort was excluded from population expansion by construction.
        candidates=[u for u,n in con.execute(f"SELECT user_id,count(*) n FROM air_views WHERE fraction>=.8 AND air_start>='{start}' AND air_end<='{cutoff}' GROUP BY 1 HAVING count(*)>=5 ORDER BY user_id").fetchall() if u not in set(targets) and int(u)%5==0]
        rng=np.random.default_rng(42);cohort=sorted(rng.choice(candidates,min(500,len(candidates)),replace=False).tolist());assert cohort
        dump(out/'reserve_cohort.json',{'users':cohort,'selection':'Non-template, user_id mod5=0, >=5 pre-cutoff events; seed42 sample up to500; no future labels inspected','cutoff':cutoff,'end':end})
        fresh=feature_data(con,start,cutoff,end,cohort,feature_cfg,vocabulary)
        preds={'selected_ranker':predict_ranker(fresh,cohort,ranker),'v4_classifier':predict_reference(fresh,cohort,classifier)}
        preds['popularity']={u:top_five(np.zeros(len(fresh['items'])),fresh['items'],fresh['pop']) for u in cohort}
        # Predictions frozen before reading this cohort's future labels.
        dump(out/'reserve_predictions.json',preds)
        truth,_=make_truth(con,cutoff,end,cohort);history=dict(zip(cohort,fresh['history'].tolist()))
        metrics={name:evaluate(truth,pred,history)[0] for name,pred in preds.items()}
        eligible=[u for u in cohort if truth[u]];assert eligible
        delta=np.array([ap_at_k(truth[u],preds['selected_ranker'][u])-ap_at_k(truth[u],preds['v4_classifier'][u]) for u in eligible])
        boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        dump(out/'reserve_evaluation.json',{'cohort_users':len(cohort),'eligible_users':len(eligible),'empty_truth_users':len(cohort)-len(eligible),'metrics':metrics,'mean_delta_vs_v4':float(delta.mean()),'paired_user_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
          'parameters_frozen_before_reserve_labels':True,'labels_used_for_fit_or_retuning':False,'caveat':'Auxiliary new-user cohort, shared calendar/items, not a fresh time period; selection audience differs from Kaggle template.'})
    # Fit twice on all allowed history, preserving the exact same chosen method.
    cutoff,end='2020-07-27 00:00:00','2020-10-19 00:00:00';checks=[]
    old_paths=['artifacts/final_submission.csv','runs/v2_candidate/submission_v2.csv','runs/v3_candidate/submission_v3.csv','runs/v4_candidate/submission_v4.csv']
    old_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old_paths}
    for repeat in range(2):
        tick=time.perf_counter();ranker,_,audit=fit_ranker(con,start,cutoff,targets,feature_cfg,vocabulary,params)
        data=feature_data(con,start,cutoff,end,targets,feature_cfg,vocabulary);pred=predict_ranker(data,targets,ranker)
        tmp=Path(f'tmp/submission_v5_repeat{repeat}.csv');check=write_submission('sweettv-tv-program-recommender/submission.csv',pred,tmp,set(data['items'].tolist()))
        legal={int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM schedule_raw WHERE filename LIKE '%31-42%' AND tv_show_id!='0'").fetchall()}
        assert all(i in legal for ids in pred.values() for i in ids)
        frozen=out/'submission_v5.csv'
        if frozen.exists():assert frozen.read_bytes()==tmp.read_bytes()
        else:frozen.write_bytes(tmp.read_bytes())
        check.update({'repeat':repeat,'fit_predict_seconds':time.perf_counter()-tick,'all_ids_in_literal_test_schedule':True,'fit_audit':audit});checks.append(check);dump(out/'submission_checks.json',checks)
    assert checks[0]['sha256']==checks[1]['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old_hashes.items())
    dump(out/'reproduction.json',{'two_full_fits_identical_sha256':checks[0]['sha256'],'all_validation_lists_identical':True,'previous_submissions_unchanged':old_hashes})
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','configs','scripts','tests'] for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
    dump(out/'source_manifest.json',manifest)
print(json.dumps({'selected':selection,'sha256':checks[0]['sha256'],'reserve':json.loads((out/'reserve_evaluation.json').read_text())},ensure_ascii=False,indent=2),flush=True)
