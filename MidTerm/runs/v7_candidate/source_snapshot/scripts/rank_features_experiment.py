import json,time
from pathlib import Path
import duckdb
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows
from sweettv.prepare import dump
from habit_features import augment_habit
from item_metadata_features import augment_item_metadata,fit_vocabulary
from lambdarank_helpers import query_groups,vector_to_scores,ranked_predictions

cfg=json.loads(Path('configs/rank_features.json').read_text());fcfg=json.loads(Path(cfg['feature_config']).read_text());rcfg=json.loads(Path(cfg['ranker_config']).read_text())
protocol=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v18_rank_features');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()];vocab=fit_vocabulary(con,cfg['vocabulary_cutoff_exclusive'],20);dump(out/'vocabulary.json',vocab)
reference=json.loads(Path('runs/v17_lambdarank/predictions.json').read_text());results=[];predictions={};audit={}
params={**rcfg['common'],'n_estimators':150,'num_leaves':15}
def features(start,anchor,end):
    data=build_features(con,start,anchor,end,targets,fcfg)
    data=augment_habit(data,con,start,anchor,targets)
    return augment_item_metadata(data,con,anchor,end,vocab)
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end'];xx=[];yy=[];gg=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,fcfg):
            data=features(start,anchor,inner_end);truth,_=make_truth(con,anchor,inner_end,targets);y,valid,_=label_rows(data,truth,targets)
            xx.append(data['x'][valid]);yy.append(y[valid]);gg.append(query_groups(data,valid));windows.append({'feature_cutoff':anchor,'label_end':inner_end})
        x=np.concatenate(xx);y=np.concatenate(yy);groups=np.concatenate(gg);del xx,yy,gg;assert x.shape[1]==86
        fits={};times={}
        for name,width in [('rank_control45',45),('rank_habit56',56),('rank_combined86',86)]:
            tick=time.perf_counter();fits[name]=(lgb.LGBMRanker(**params).fit(x[:,:width],y,group=groups),width);times[name]=time.perf_counter()-tick
        audit[split['name']]={'inner_windows':windows,'outer_labels_used_in_fit':False,'fit_seconds':times,'query_count':len(groups),'group_sum_equals_rows':bool(groups.sum()==len(y))};dump(out/'fit_audit.json',audit);del x,y,groups
        data=features(start,cutoff,end);truth,_=make_truth(con,cutoff,end,targets);preds={}
        for name,(model,width) in fits.items():
            score=vector_to_scores(data,targets,model.booster_.predict(data['x'][:,:width]),True);pred=ranked_predictions(data,targets,score)
            if name=='rank_control45':assert pred==reference[split['name']]['predictions']['lambdarank150_leaf15']
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())));preds[name]=pred;results.append({'split':split['name'],'method':name,**metrics});dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions);dump(out/'feature_names.json',data['feature_names'].tolist())
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name];summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
