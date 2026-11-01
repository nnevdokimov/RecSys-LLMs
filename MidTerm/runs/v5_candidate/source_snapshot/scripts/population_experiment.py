import json
from pathlib import Path
import duckdb
import numpy as np
from threadpoolctl import threadpool_limits
from sweettv.experiment import evaluate,make_truth
from sweettv.nested_rerank import build_features
from sweettv.prepare import dump
from population_helpers import fit_population,population_predictions

cfg=json.loads(Path('configs/population.json').read_text());feature_cfg=json.loads(Path(cfg['feature_config']).read_text())
protocol=json.loads(Path('configs/advanced.json').read_text());out=Path('runs/v13_population');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
con=duckdb.connect('artifacts/data.duckdb',read_only=True);con.execute('SET threads=4')
targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
old=json.loads(Path('runs/v10_category/predictions.json').read_text());results=[];predictions={};audit={}
with threadpool_limits(limits=4):
    for split in protocol['splits']:
        start,cutoff,end=protocol['train_start'],split['cutoff'],split['end']
        reference,augmented,audit[split['name']]=fit_population(con,start,cutoff,targets,feature_cfg,cfg)
        dump(out/'fit_audit.json',audit)
        data=build_features(con,start,cutoff,end,targets,feature_cfg)
        preds=population_predictions(data,targets,reference,augmented)
        assert preds['template_only_reference']==old[split['name']]['predictions']['nested_hgb_weight1.0']
        truth,_=make_truth(con,cutoff,end,targets)
        for name,pred in preds.items():
            metrics,_=evaluate(truth,pred,dict(zip(targets,data['history'].tolist())))
            results.append({'split':split['name'],'method':name,**metrics});dump(out/'results.json',results)
            print(split['name'],name,f'{metrics["map5"]:.6f}',flush=True)
        predictions[split['name']]={'truth':truth,'predictions':preds};dump(out/'predictions.json',predictions)
summary=[]
for name in sorted({r['method'] for r in results}):
    rows=[r for r in results if r['method']==name]
    summary.append({'method':name,'mean_map5':float(np.mean([r['map5'] for r in rows])),'split_scores':{r['split']:r['map5'] for r in rows}})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary)
print('SUMMARY',json.dumps(summary,indent=2),flush=True)
