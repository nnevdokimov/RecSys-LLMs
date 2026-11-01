"""Preserve all historical search rows, including repeated/superseded references."""
import csv,json
from pathlib import Path
base=Path('runs/v4_candidate/full_search_summary.csv')
with base.open(newline='') as f:rows=list(csv.DictReader(f))
phases=['v11_capacity','v12_partial','v13_population','v14_habit','v15_item_metadata','v16_frequency','v17_lambdarank','v18_rank_features','v19_rank_final']
for phase in phases:
    for r in json.loads((Path('runs')/phase/'summary.json').read_text()):
        rows.append({'phase':phase,'method':r['method'],'mean_map5':r['mean_map5'],**r['split_scores'],'validity':'historical_selection; shared_overlapping_splits'})
assert len(rows)==165,len(rows)
p=Path('runs/v5_candidate/full_search_summary.csv')
with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['phase','method','mean_map5','early_10w','original_10w','late_6w','validity']);w.writeheader();w.writerows(rows)
print(p,len(rows))
