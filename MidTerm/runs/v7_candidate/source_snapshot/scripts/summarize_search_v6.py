"""Preserve all historical search rows, including repeated/superseded references."""
import csv,json
from pathlib import Path
base=Path('runs/v5_candidate/full_search_summary.csv')
with base.open(newline='') as f:rows=list(csv.DictReader(f))
phases=['v20_rank_fusion','v21_rank_capacity','v22_rank_objective','v23_checkpoint_fusion']
for phase in phases:
    for r in json.loads((Path('runs')/phase/'summary.json').read_text()):
        rows.append({'phase':phase,'method':r['method'],'mean_map5':r['mean_map5'],**r['split_scores'],'validity':'historical_selection; shared_overlapping_splits'})
assert len(rows)==182,len(rows)
p=Path('runs/v6_candidate/full_search_summary.csv')
with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['phase','method','mean_map5','early_10w','original_10w','late_6w','validity']);w.writeheader();w.writerows(rows)
print(p,len(rows))
