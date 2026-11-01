"""Bounded rank-list ensemble comparison; historical validation only."""
import json
from pathlib import Path
import numpy as np
import duckdb
from sweettv.prepare import dump
from sweettv.metrics import ap_at_k
from sweettv.models import Models,top_five
from sweettv.experiment import train_rows,load_metadata,candidates_for

cfg=json.loads(Path('configs/rank_fusion.json').read_text());protocol=json.loads(Path('configs/advanced.json').read_text())
out=Path('runs/v20_rank_fusion');out.mkdir(exist_ok=True);dump(out/'config_snapshot.json',cfg)
rank=json.loads(Path('runs/v19_rank_final/predictions.json').read_text());hgb=json.loads(Path('runs/v10_category/predictions.json').read_text())
con=duckdb.connect('artifacts/data.duckdb',read_only=True);targets=[r[0] for r in con.execute('SELECT user_id FROM template').fetchall()];results=[];preds={}
for split in protocol['splits']:
    name=split['name'];entry=rank[name];truth=entry['truth'];base=entry['predictions'][cfg['base']]
    items=candidates_for(con,split['cutoff'],split['end']);rows=train_rows(con,protocol['train_start'],split['cutoff'])
    users=sorted({r['user_id'] for r in rows}|set(targets));m=Models(rows,load_metadata(con,split['cutoff']),users,items,split['cutoff'],targets)
    pop=m.pop;items=m.candidates;preds[name]={'truth':truth,'predictions':{}}
    for partner in cfg['partners']:
        other=hgb[name]['predictions']['nested_hgb_weight1.0'] if partner=='v4_hgb' else entry['predictions'][partner]
        for w in cfg['partner_weights']:
            method=f'{partner}_weight{w}';pred={}
            for u in targets:
                score={}
                for weight,ls in [(1-w,base[u]),(w,other[u])]:
                    for r,i in enumerate(ls,1):score[i]=score.get(i,0)+weight/(cfg['rank_offset']+r)
                vec=np.array([score.get(int(i),0) for i in items]);pred[u]=top_five(vec,items,pop)
            vals=[ap_at_k(truth[u],pred[u]) for u in targets if truth[u]]
            results.append({'method':method,'split':name,'map5':float(np.mean(vals))});preds[name]['predictions'][method]=pred
summary=[]
for method in sorted({r['method'] for r in results}):
    ss={r['split']:r['map5'] for r in results if r['method']==method};summary.append({'method':method,'mean_map5':float(np.mean(list(ss.values()))),'split_scores':ss})
summary.sort(key=lambda r:-r['mean_map5']);dump(out/'summary.json',summary);dump(out/'results.json',results);dump(out/'predictions.json',preds)
print(json.dumps(summary,indent=2))
