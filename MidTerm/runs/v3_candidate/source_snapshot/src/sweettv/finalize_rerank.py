"""Freeze a validation-selected nested reranker, reproduce it, and write v3 CSV."""
import hashlib
import json
import time
from pathlib import Path

import duckdb
from threadpoolctl import threadpool_limits

from .experiment import write_submission
from .nested_rerank import build_features,fit_before,predict
from .prepare import dump


def run():
    out=Path('runs/v3_candidate');out.mkdir(parents=True,exist_ok=True)
    choices=[]
    for folder,config in [('runs/v8_nested','configs/nested_rerank.json'),('runs/v9_nested_wide','configs/nested_wide.json')]:
        for row in json.loads((Path(folder)/'summary.json').read_text()):
            choices.append({**row,'folder':folder,'config':config})
    selected=max(choices,key=lambda r:r['mean_map5'])
    assert selected['mean_map5']>0.40964937936445184,'Nested reranker did not beat recent-interest reference'
    weight=float(selected['method'].split('weight')[-1]);cfg=json.loads(Path(selected['config']).read_text())
    selection={**selected,'classifier_score_weight':weight,'criterion':'Highest mean across three historical rolling splits; same overlap and selection-bias caveats',
               'private_score_used_for_parameter_selection':False,'private_score':None,'parameters':cfg}
    path=out/'selection.json'
    if path.exists():assert json.loads(path.read_text())==selection,'Frozen selection changed'
    else:dump(path,selection)
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','configs'] for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
    dump(out/'source_manifest.json',manifest)
    con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
    targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    start,cutoff,end='2020-03-09 00:00:00','2020-07-27 00:00:00','2020-10-19 00:00:00'
    with threadpool_limits(limits=4):
        # Refit the original hold-out once and compare all users' exact lists.
        val_cut,val_end='2020-05-18 00:00:00','2020-07-27 00:00:00'
        clf,audit=fit_before(con,start,val_cut,targets,cfg)
        validation=build_features(con,start,val_cut,val_end,targets,cfg)
        actual=predict(validation,targets,clf,weight)
        expected=json.loads((Path(selected['folder'])/'predictions.json').read_text())['original_10w']['predictions'][selected['method']]
        assert actual==expected,'Selected validation predictions failed reproduction'
        dump(out/'validation_reproduction.json',{'all_1260_ranked_lists_identical':True,'fit_audit':audit})
        # Freeze and reproduce full allowed-history fit twice, without test views.
        baseline_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('artifacts/final_submission.csv'),Path('runs/v2_candidate/submission_v2.csv')]}
        checks=[]
        for repeat in range(2):
            tick=time.perf_counter();clf,audit=fit_before(con,start,cutoff,targets,cfg)
            data=build_features(con,start,cutoff,end,targets,cfg)
            pred=predict(data,targets,clf,weight)
            temp=Path(f'tmp/submission_v3_repeat{repeat}.csv')
            check=write_submission('sweettv-tv-program-recommender/submission.csv',pred,temp,set(data['items'].tolist()))
            literal_test={int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM schedule_raw WHERE filename LIKE '%31-42%' AND tv_show_id!='0'").fetchall()}
            assert all(item in literal_test for ids in pred.values() for item in ids)
            check.update({'repeat':repeat,'fit_predict_seconds':time.perf_counter()-tick,'all_ids_in_literal_test_schedule':True,'fit_audit':audit})
            candidate=out/'submission_v3.csv'
            if candidate.exists():assert candidate.read_bytes()==temp.read_bytes(),'Frozen submission differs'
            else:candidate.write_bytes(temp.read_bytes())
            checks.append(check);dump(out/'submission_checks.json',checks)
        assert checks[0]['sha256']==checks[1]['sha256']
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in baseline_hashes.items())
        dump(out/'reproduction.json',{'two_full_fits_identical_sha256':checks[0]['sha256'],'previous_submissions_unchanged':baseline_hashes,'validation_all_ranked_lists_identical':True})
    print(json.dumps({'selection':selection,'submission_sha256':checks[0]['sha256'],'path':str(candidate)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
