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

out=Path('runs/v6_candidate');out.mkdir(exist_ok=True)
cfg=json.loads(Path('configs/rank_capacity.json').read_text());rank_cfg=json.loads(Path(cfg['ranker_config']).read_text());feature_cfg=json.loads(Path(cfg['feature_config']).read_text())
selected=json.loads(Path('runs/v21_rank_capacity/summary.json').read_text())[0]
assert selected['mean_map5']>=0.42020715673434345-1e-12
variant=next(v for v in cfg['variants'] if v['name']==selected['method'])
params={**rank_cfg['common'],'n_estimators':150,**{k:v for k,v in variant.items() if k!='name'}}
if params['objective']=='rank_xendcg':params.pop('lambdarank_truncation_level')
vocabulary=json.loads(Path('runs/v21_rank_capacity/vocabulary.json').read_text())
selection={**selected,'source_folder':'runs/v21_rank_capacity','parameters':params,'feature_config':cfg['feature_config'],'feature_count':86,'vocabulary':vocabulary,
 'criterion':'Highest mean across predefined three historical rolling splits; broad search and overlapping windows disclosed',
 'reserve_user_labels_used_for_choice':False,'private_score_used_for_parameter_selection':False,'private_score':None,'reserve_audit_for_this_version':False,'caveat':'Mean gain over v5 is only 0.00008774; late split is lower; no independent confirmation'}
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
    expected=json.loads(Path('runs/v21_rank_capacity/predictions.json').read_text())['original_10w']['predictions'][selected['method']]
    expected_old=json.loads(Path('runs/v10_category/predictions.json').read_text())['original_10w']['predictions']['nested_hgb_weight1.0']
    assert new==expected and old==expected_old
    dump(out/'validation_reproduction.json',{'new_all_1260_lists_identical':True,'reference_all_1260_lists_identical':True,'fit_audit':audit})
    # Reserve cohort was audited for v5; it is not reused to select or retune v6.
    # Fit twice on all allowed history, preserving the exact same chosen method.
    cutoff,end='2020-07-27 00:00:00','2020-10-19 00:00:00';checks=[]
    old_paths=['artifacts/final_submission.csv','runs/v2_candidate/submission_v2.csv','runs/v3_candidate/submission_v3.csv','runs/v4_candidate/submission_v4.csv','runs/v5_candidate/submission_v5.csv']
    old_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old_paths}
    for repeat in range(2):
        tick=time.perf_counter();ranker,_,audit=fit_ranker(con,start,cutoff,targets,feature_cfg,vocabulary,params)
        data=feature_data(con,start,cutoff,end,targets,feature_cfg,vocabulary);pred=predict_ranker(data,targets,ranker)
        tmp=Path(f'tmp/submission_v6_repeat{repeat}.csv');check=write_submission('sweettv-tv-program-recommender/submission.csv',pred,tmp,set(data['items'].tolist()))
        legal={int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM schedule_raw WHERE filename LIKE '%31-42%' AND tv_show_id!='0'").fetchall()}
        assert all(i in legal for ids in pred.values() for i in ids)
        frozen=out/'submission_v6.csv'
        if frozen.exists():assert frozen.read_bytes()==tmp.read_bytes()
        else:frozen.write_bytes(tmp.read_bytes())
        check.update({'repeat':repeat,'fit_predict_seconds':time.perf_counter()-tick,'all_ids_in_literal_test_schedule':True,'fit_audit':audit});checks.append(check);dump(out/'submission_checks.json',checks)
    assert checks[0]['sha256']==checks[1]['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old_hashes.items())
    dump(out/'reproduction.json',{'two_full_fits_identical_sha256':checks[0]['sha256'],'all_validation_lists_identical':True,'previous_submissions_unchanged':old_hashes})
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','configs','scripts','tests'] for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
    dump(out/'source_manifest.json',manifest)
print(json.dumps({'selected':selection,'sha256':checks[0]['sha256']},ensure_ascii=False,indent=2),flush=True)
