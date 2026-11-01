"""Compare equal and frequency-derived gains using inner labels only."""
import json,time
from pathlib import Path
import duckdb
import numpy as np
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from final_ranker import feature_data
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import inner_windows,label_rows
from sweettv.prepare import dump
from lambdarank_helpers import query_groups,vector_to_scores,ranked_predictions

cfg=json.loads(Path('configs/rank_graded.json').read_text());selection=json.loads(Path('runs/v5_candidate/selection.json').read_text());fcfg=json.loads(Path(selection['feature_config']).read_text());protocol=json.loads(Path('configs/advanced.json').read_text());vocab=selection['vocabulary']
out=Path('runs/v25_rank_graded');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4');targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
reference=json.loads(Path('runs/v19_rank_final/predictions.json').read_text());results=[];predictions={};audits={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end'];xx=[];ys={n:[] for n in cfg['variants']};gg=[];windows=[]
        for anchor,inner_end in inner_windows(start,cutoff,fcfg):
            data=feature_data(con,start,anchor,inner_end,targets,fcfg,vocab);truth,counts=make_truth(con,anchor,inner_end,targets);y,valid,_=label_rows(data,truth,targets)
            gains_rank={u:{item:5-j for j,item in enumerate(truth[u])} for u in targets};gains_count={u:{item:min(5,n) for item,n in zip(truth[u],counts[u])} for u in targets}
            ranked=np.fromiter((gains_rank[targets[r]].get(int(data['items'][c]),0) for r,c in zip(data['rr'],data['cc'])),dtype=np.int8)
            counted=np.fromiter((gains_count[targets[r]].get(int(data['items'][c]),0) for r,c in zip(data['rr'],data['cc'])),dtype=np.int8)
            assert np.array_equal(y>0,ranked>0) and np.array_equal(y>0,counted>0)
            xx.append(data['x'][valid]);gg.append(query_groups(data,valid))
            for name,labels in [('binary_reference',y),('top5_rank_gain',ranked),('top5_count_cap5',counted)]:ys[name].append(labels[valid])
            windows.append({'feature_cutoff':anchor,'label_end':inner_end})
        x=np.concatenate(xx);groups=np.concatenate(gg);del xx,gg;fits={};times={}
        for name in cfg['variants']:
            y=np.concatenate(ys[name]);params=dict(selection['parameters'])
            if name!='binary_reference':params['label_gain']=list(range(6))
            tick=time.perf_counter();fits[name]=lgb.LGBMRanker(**params).fit(x,y,group=groups);times[name]=time.perf_counter()-tick
        del x,y,ys,groups
        audits[split['name']]={'inner_windows':windows,'outer_labels_used_in_fit':False,'evaluation_truth_stays_binary':True,'fit_seconds':times};dump(out/'fit_audit.json',audits)
        data=feature_data(con,start,cutoff,end,targets,fcfg,vocab);truth,_=make_truth(con,cutoff,end,targets);preds={}
        for name,model in fits.items():
            pred=ranked_predictions(data,targets,vector_to_scores(data,targets,model.booster_.predict(data['x']),True))
            if name=='binary_reference':assert pred==reference[split['name']]['predictions']['combined86_reference']
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())));preds[name]=pred;results.append({'split':split['name'],'method':name,**metrics});dump(out/'results.json',results);print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
summary=[]
for name in cfg['variants']:
    rows=[r for r in results if r['method']==name];summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
