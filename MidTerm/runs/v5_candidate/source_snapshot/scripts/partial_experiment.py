"""Ablation: expanded partial candidates, with/without partial feature columns."""
import json,time
from pathlib import Path
import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows,predict
from sweettv.prepare import dump

cfg=json.loads(Path('configs/nested_partial.json').read_text());protocol=json.loads(Path('configs/advanced.json').read_text())
out=Path('runs/v12_partial');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
results=[];predictions={};audit={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end'];xx=[];yy=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,cfg):
            data=build_features(con,start,anchor,inner_end,targets,cfg);truth,_=make_truth(con,anchor,inner_end,targets)
            y,valid,coverage=label_rows(data,truth,targets);xx.append(data['x'][valid]);yy.append(y[valid])
            windows.append({'feature_cutoff':anchor,'label_end':inner_end,'candidate_gt_coverage':coverage})
        x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy
        mask=np.array(['partial' not in str(n) for n in data['feature_names']]);assert mask.sum()==45 and len(mask)==55
        fits={};times={}
        for name,columns in [('partial_candidates_control45',mask),('partial_full55',np.ones(len(mask),bool))]:
            tick=time.perf_counter();clf=HistGradientBoostingClassifier(**cfg['estimator'],random_state=cfg['seed'])
            clf.fit(x[:,columns],y);fits[name]=(clf,columns);times[name]=time.perf_counter()-tick
        del x,y
        audit[split['name']]={'outer_cutoff':cutoff,'inner_windows':windows,'outer_labels_used_in_fit':False,'fit_seconds':times}
        dump(out/'fit_audit.json',audit)
        data=build_features(con,start,cutoff,end,targets,cfg);truth,_=make_truth(con,cutoff,end,targets);preds={}
        _,_,coverage=label_rows(data,truth,targets)
        for name,(clf,columns) in fits.items():
            pred=predict({**data,'x':data['x'][:,columns]},targets,clf,1.0)
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())))
            preds[name]=pred;results.append({'split':split['name'],'method':name,**metrics,'fit_seconds':times[name],'candidate_gt_coverage':coverage})
            dump(out/'results.json',results);print(split['name'],name,f'{metrics["map5"]:.6f}','coverage',round(coverage,4),flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
        dump(out/'feature_names.json',data['feature_names'].tolist())
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name]
    summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
