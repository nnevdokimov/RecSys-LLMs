import json,time
from pathlib import Path
import duckdb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows
from sweettv.prepare import dump
from sweettv.models import row_max_norm,top_five
from habit_features import augment_habit

cfg=json.loads(Path('configs/habit.json').read_text());fcfg=json.loads(Path(cfg['feature_config']).read_text())
protocol=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v14_habit');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()];reference=json.loads(Path('runs/v10_category/predictions.json').read_text())
results=[];predictions={};audit={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end'];xx=[];yy=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,fcfg):
            data=augment_habit(build_features(con,start,anchor,inner_end,targets,fcfg),con,start,anchor,targets)
            truth,_=make_truth(con,anchor,inner_end,targets);y,valid,_=label_rows(data,truth,targets)
            xx.append(data['x'][valid]);yy.append(y[valid]);windows.append({'feature_cutoff':anchor,'label_end':inner_end})
        x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy;fits=[];times=[]
        for width in [45,56]:
            tick=time.perf_counter();clf=HistGradientBoostingClassifier(**fcfg['estimator'],random_state=fcfg['seed']).fit(x[:,:width],y)
            fits.append(clf);times.append(time.perf_counter()-tick)
        del x,y
        audit[split['name']]={'inner_windows':windows,'outer_labels_used_in_fit':False,'fit_seconds':times};dump(out/'fit_audit.json',audit)
        data=augment_habit(build_features(con,start,cutoff,end,targets,fcfg),con,start,cutoff,targets)
        raw=[]
        for width,clf in zip([45,56],fits):
            score=np.zeros((len(targets),len(data['items'])));score[data['rr'],data['cc']]=clf.predict_proba(data['x'][:,:width])[:,1];raw.append(row_max_norm(score))
        truth,_=make_truth(con,cutoff,end,targets);preds={}
        for name,score in [('habit_control45',raw[0]),('habit_full56',raw[1]),('habit_fusion50',.5*raw[0]+.5*raw[1])]:
            score=score.copy();score[data['history']==0]=0
            pred={u:top_five(score[i],data['items'],data['pop']) for i,u in enumerate(targets)}
            if name=='habit_control45':assert pred==reference[split['name']]['predictions']['nested_hgb_weight1.0']
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())));preds[name]=pred
            results.append({'split':split['name'],'method':name,**metrics});dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
        dump(out/'feature_names.json',data['feature_names'].tolist())
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name];summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
