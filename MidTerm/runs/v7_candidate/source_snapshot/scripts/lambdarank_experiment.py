import json,time
from pathlib import Path
import duckdb
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows
from sweettv.prepare import dump
from lambdarank_helpers import query_groups,vector_to_scores,ranked_predictions

cfg=json.loads(Path('configs/lambdarank.json').read_text());fcfg=json.loads(Path(cfg['feature_config']).read_text())
protocol=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v17_lambdarank');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
dump(out/'environment.json',{'lightgbm':lgb.__version__,'libomp_preexisting':True})
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()];reference=json.loads(Path('runs/v10_category/predictions.json').read_text())
results=[];predictions={};audit={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end'];xx=[];yy=[];gg=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,fcfg):
            data=build_features(con,start,anchor,inner_end,targets,fcfg)
            truth,_=make_truth(con,anchor,inner_end,targets);y,valid,_=label_rows(data,truth,targets)
            xx.append(data['x'][valid]);yy.append(y[valid]);gg.append(query_groups(data,valid));windows.append({'feature_cutoff':anchor,'label_end':inner_end})
        x=np.concatenate(xx);y=np.concatenate(yy);group=np.concatenate(gg);del xx,yy,gg
        assert group.sum()==len(y);fits={};times={};tick=time.perf_counter()
        classifier=HistGradientBoostingClassifier(**fcfg['estimator'],random_state=fcfg['seed']).fit(x,y);times['reference']=time.perf_counter()-tick
        for variant in cfg['variants']:
            params={**cfg['common'],**{k:v for k,v in variant.items() if k!='name'}};tick=time.perf_counter()
            ranker=lgb.LGBMRanker(**params).fit(x,y,group=group)
            fits[variant['name']]=ranker;times[variant['name']]=time.perf_counter()-tick
        audit[split['name']]={'inner_windows':windows,'outer_labels_used_in_fit':False,'query_count':len(group),'training_rows':len(y),'group_sum_equals_rows':True,'fit_seconds':times}
        dump(out/'fit_audit.json',audit);del x,y,group
        data=build_features(con,start,cutoff,end,targets,fcfg)
        ref=vector_to_scores(data,targets,classifier.predict_proba(data['x'])[:,1]);scores={'lambdarank_classifier_reference':ref}
        for name,model in fits.items():
            scores[name]=vector_to_scores(data,targets,model.booster_.predict(data['x']),True)
            scores[name+'__fusion50']=.5*ref+.5*scores[name]
        truth,_=make_truth(con,cutoff,end,targets);preds={}
        for name,score in scores.items():
            pred=ranked_predictions(data,targets,score)
            if name=='lambdarank_classifier_reference':assert pred==reference[split['name']]['predictions']['nested_hgb_weight1.0']
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())));preds[name]=pred
            results.append({'split':split['name'],'method':name,**metrics});dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name];summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
