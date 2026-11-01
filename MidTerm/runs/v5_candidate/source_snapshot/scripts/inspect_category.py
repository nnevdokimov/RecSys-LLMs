"""Attach final nested predictions to the SAME preselected two user examples."""
import copy
import json
from pathlib import Path

import duckdb

from sweettv.experiment import load_metadata
from sweettv.metrics import ap_at_k
from sweettv.prepare import dump

out=Path('runs/v4_candidate')
selected=json.loads((out/'selection.json').read_text())
source=json.loads((Path(selected['folder'])/'predictions.json').read_text())['original_10w']
recent=json.loads(Path('runs/v7_soft_trend/predictions.json').read_text())['original_10w']
examples=copy.deepcopy(json.loads(Path('runs/v2_candidate/user_examples.json').read_text()))
con=duckdb.connect('artifacts/data.duckdb',read_only=True)
# Post-prediction display only: this lookup never feeds scoring or fit.
metadata=load_metadata(con,'2020-07-27 00:00:00')
for user in examples:
    uid=user['user_id'];truth=source['truth'][uid]
    for name,ids in [('recent_trend',recent['predictions']['trend28'][uid]),('nested_final',source['predictions'][selected['method']][uid])]:
        user['recommendations'][name]=[{'id':int(i),'title':metadata.get(i,{}).get('tv_show_title','Нет названия в разрешённом расписании'),
          'category':metadata.get(i,{}).get('tv_show_category','Нет категории'),'hit':i in truth} for i in ids]
        user['ap5'][name]=ap_at_k(truth,ids)
    user['display_metadata_note']='Looked up after predictions, through validation end, for report display only.'
dump(out/'user_examples.json',examples)
print(json.dumps([{'user_id':u['user_id'],'final_top5':u['recommendations']['nested_final'],'ap5':u['ap5']} for u in examples],ensure_ascii=False,indent=2))
