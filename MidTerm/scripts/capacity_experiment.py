"""Controlled capacity check, reusing exactly the already-validated 45 features."""
import json
import time
from pathlib import Path

import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows,predict
from sweettv.prepare import dump

cfg=json.loads(Path('configs/capacity.json').read_text());feature_cfg=json.loads(Path(cfg['feature_config']).read_text())
protocol=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v11_capacity');out.mkdir(exist_ok=True)
dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
reference=json.loads(Path('runs/v10_category/predictions.json').read_text())
results=[];predictions={};audit={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end']
        xx=[];yy=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,feature_cfg):
            data=build_features(con,start,anchor,inner_end,targets,feature_cfg)
            truth,_=make_truth(con,anchor,inner_end,targets);y,valid,coverage=label_rows(data,truth,targets)
            xx.append(data['x'][valid]);yy.append(y[valid]);windows.append({'feature_cutoff':anchor,'label_end':inner_end})
        x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy
        fits={};times={}
        for variant in [cfg['reference']]+cfg['variants']:
            params={**feature_cfg['estimator'],**{k:v for k,v in variant.items() if k!='name'}}
            tick=time.perf_counter();clf=HistGradientBoostingClassifier(**params,random_state=feature_cfg['seed'])
            clf.fit(x,y);fits[variant['name']]=clf;times[variant['name']]=time.perf_counter()-tick
            print('FIT',split['name'],variant['name'],round(times[variant['name']],2),flush=True)
        del x,y
        audit[split['name']]={'inner_windows':windows,'all_fit_label_ends_at_or_before_outer_cutoff':all(w['label_end']<=cutoff for w in windows),'outer_labels_used_in_fit':False,'fit_seconds':times}
        dump(out/'fit_audit.json',audit)
        data=build_features(con,start,cutoff,end,targets,feature_cfg)
        truth,_=make_truth(con,cutoff,end,targets);preds={}
        for name,clf in fits.items():
            pred=predict(data,targets,clf,1.0);metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())))
            if name=='hgb100_leaf15':assert pred==reference[split['name']]['predictions']['nested_hgb_weight1.0']
            preds[name]=pred;results.append({'split':split['name'],'method':name,**metrics,'fit_seconds':times[name]})
            dump(out/'results.json',results);print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name]
    summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
