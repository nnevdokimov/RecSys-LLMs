"""Freeze, reproduce and validate the historical graded-gain winner."""
import hashlib,json,time
from pathlib import Path
import duckdb
from threadpoolctl import threadpool_limits
from final_ranker import feature_data,predict_ranker
from graded_ranker import fit_graded_ranker
from sweettv.experiment import write_submission
from sweettv.prepare import dump

out=Path('runs/v7_candidate');out.mkdir(exist_ok=True)
base=json.loads(Path('runs/v5_candidate/selection.json').read_text());winner=json.loads(Path('runs/v25_rank_graded/summary.json').read_text())[0]
assert winner['method']=='top5_rank_gain' and winner['mean_map5']>json.loads(Path('runs/v6_candidate/selection.json').read_text())['mean_map5']
params={**base['parameters'],'label_gain':list(range(6))}
selection={**winner,'parameters':params,'vocabulary':base['vocabulary'],'feature_config':base['feature_config'],'feature_count':86,
 'source_folder':'runs/v25_rank_graded','label_strategy':'Top5 gains5..1 ordered by inner future frequency, zeros outside Top5; external MAP stays binary',
 'criterion':'Highest historical three-split mean; broad overlapping search disclosed','private_score_used_for_parameter_selection':False,
 'reserve_user_labels_used_for_choice':False,'private_score':None,'caveat':'Mean gain0.000666 over v5, late split lower, no independent significance claim'}
if (out/'selection.json').exists():assert json.loads((out/'selection.json').read_text())==selection
else:dump(out/'selection.json',selection)
feature_cfg=json.loads(Path(selection['feature_config']).read_text());con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()];start='2020-03-09 00:00:00'
with threadpool_limits(limits=4):
    cutoff,end='2020-05-18 00:00:00','2020-07-27 00:00:00'
    model,_,audit=fit_graded_ranker(con,start,cutoff,targets,feature_cfg,selection['vocabulary'],params)
    data=feature_data(con,start,cutoff,end,targets,feature_cfg,selection['vocabulary']);pred=predict_ranker(data,targets,model)
    expected=json.loads(Path('runs/v25_rank_graded/predictions.json').read_text())['original_10w']['predictions'][winner['method']]
    assert pred==expected;dump(out/'validation_reproduction.json',{'all_1260_lists_identical':True,'fit_audit':audit})
    previous=['artifacts/final_submission.csv']+[f'runs/v{v}_candidate/submission_v{v}.csv' for v in [2,3,4,5,6]]
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in previous};checks=[]
    cutoff,end='2020-07-27 00:00:00','2020-10-19 00:00:00'
    for repeat in range(2):
        tick=time.perf_counter();model,_,audit=fit_graded_ranker(con,start,cutoff,targets,feature_cfg,selection['vocabulary'],params)
        data=feature_data(con,start,cutoff,end,targets,feature_cfg,selection['vocabulary']);pred=predict_ranker(data,targets,model)
        tmp=Path(f'tmp/submission_v7_repeat{repeat}.csv');check=write_submission('sweettv-tv-program-recommender/submission.csv',pred,tmp,set(data['items'].tolist()))
        legal={int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM schedule_raw WHERE filename LIKE '%31-42%' AND tv_show_id!='0'").fetchall()}
        assert all(i in legal for ids in pred.values() for i in ids)
        frozen=out/'submission_v7.csv'
        if frozen.exists():assert frozen.read_bytes()==tmp.read_bytes()
        else:frozen.write_bytes(tmp.read_bytes())
        check.update({'repeat':repeat,'fit_predict_seconds':time.perf_counter()-tick,'all_ids_in_literal_test_schedule':True,'fit_audit':audit});checks.append(check);dump(out/'submission_checks.json',checks)
    assert checks[0]['sha256']==checks[1]['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    dump(out/'reproduction.json',{'two_full_fits_identical_sha256':checks[0]['sha256'],'all_validation_lists_identical':True,'previous_submissions_unchanged':hashes})
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','configs','scripts','tests'] for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in str(p)};dump(out/'source_manifest.json',manifest)
print(json.dumps({'selection':selection,'sha256':checks[0]['sha256']},ensure_ascii=False,indent=2),flush=True)
