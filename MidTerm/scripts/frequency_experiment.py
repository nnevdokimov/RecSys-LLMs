import json,time
from pathlib import Path
import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows
from sweettv.prepare import dump
from sweettv.models import row_max_norm,top_five
from frequency_targets import frequency_share_targets

cfg=json.loads(Path('configs/frequency_loss.json').read_text());fcfg=json.loads(Path(cfg['feature_config']).read_text())
protocol=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v16_frequency');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()];reference=json.loads(Path('runs/v10_category/predictions.json').read_text())
results=[];predictions={};audit={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end'];xx=[];yb=[];yf=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,fcfg):
            data=build_features(con,start,anchor,inner_end,targets,fcfg)
            truth,_=make_truth(con,anchor,inner_end,targets);y,valid,_=label_rows(data,truth,targets)
            freq=frequency_share_targets(data,con,anchor,inner_end,targets)
            xx.append(data['x'][valid]);yb.append(y[valid]);yf.append(freq[valid]);windows.append({'feature_cutoff':anchor,'label_end':inner_end})
        x=np.concatenate(xx);binary=np.concatenate(yb);freq=np.concatenate(yf);del xx,yb,yf
        kwargs={**fcfg['estimator'],'random_state':fcfg['seed']};tick=time.perf_counter()
        classifier=HistGradientBoostingClassifier(**kwargs).fit(x,binary)
        regressor=HistGradientBoostingRegressor(**kwargs,loss='poisson').fit(x,freq)
        audit[split['name']]={'inner_windows':windows,'outer_labels_used_in_fit':False,'fit_seconds':time.perf_counter()-tick,'frequency_target_min':float(freq.min()),'frequency_target_max':float(freq.max())}
        dump(out/'fit_audit.json',audit);del x,binary,freq
        data=build_features(con,start,cutoff,end,targets,fcfg);raw=[]
        for vector in [classifier.predict_proba(data['x'])[:,1],regressor.predict(data['x'])]:
            assert np.isfinite(vector).all() and np.all(vector>=0)
            score=np.zeros((len(targets),len(data['items'])));score[data['rr'],data['cc']]=vector;raw.append(row_max_norm(score))
        truth,_=make_truth(con,cutoff,end,targets);preds={}
        for name,score in [('frequency_classifier_reference',raw[0]),('frequency_poisson',raw[1]),('frequency_fusion50',.5*raw[0]+.5*raw[1])]:
            score=score.copy();score[data['history']==0]=0
            pred={u:top_five(score[i],data['items'],data['pop']) for i,u in enumerate(targets)}
            if name=='frequency_classifier_reference':assert pred==reference[split['name']]['predictions']['nested_hgb_weight1.0']
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())));preds[name]=pred
            results.append({'split':split['name'],'method':name,**metrics});dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name];summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
