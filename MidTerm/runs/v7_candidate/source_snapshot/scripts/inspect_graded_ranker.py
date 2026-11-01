import copy,json
from pathlib import Path
import duckdb
from sweettv.experiment import load_metadata
from sweettv.metrics import ap_at_k
from sweettv.prepare import dump

out=Path('runs/v7_candidate');selection=json.loads((out/'selection.json').read_text())
source=json.loads((Path(selection['source_folder'])/'predictions.json').read_text())['original_10w']
examples=copy.deepcopy(json.loads(Path('runs/v5_candidate/user_examples.json').read_text()))
con=duckdb.connect('artifacts/data.duckdb',read_only=True);meta=load_metadata(con,'2020-07-27 00:00:00')
for user in examples:
    user['recommendations']['v5_rank']=user['recommendations'].pop('rank_final');user['ap5']['v5_rank']=user['ap5'].pop('rank_final')
    uid=user['user_id'];ids=source['predictions'][selection['method']][uid];truth=source['truth'][uid]
    user['recommendations']['rank_final']=[{'id':int(i),'title':meta.get(i,{}).get('tv_show_title','Нет названия'),'category':meta.get(i,{}).get('tv_show_category','Нет категории'),'hit':i in truth} for i in ids]
    user['ap5']['rank_final']=ap_at_k(truth,ids)
dump(out/'user_examples.json',examples)
print(json.dumps([{'user_id':u['user_id'],'top5':u['recommendations']['rank_final'],'ap5':u['ap5']['rank_final']} for u in examples],ensure_ascii=False,indent=2))
