"""Inner-window rank gains for the frozen 86-feature programme ranker."""
import time
import numpy as np
import lightgbm as lgb
from final_ranker import feature_data
from sweettv.experiment import make_truth
from sweettv.nested_rerank import inner_windows,label_rows
from lambdarank_helpers import query_groups


def fit_graded_ranker(con,start,cutoff,targets,feature_cfg,vocabulary,params):
    xx=[];yy=[];gg=[];audit=[]
    for anchor,end in inner_windows(start,cutoff,feature_cfg):
        data=feature_data(con,start,anchor,end,targets,feature_cfg,vocabulary)
        truth,_=make_truth(con,anchor,end,targets);binary,valid,_=label_rows(data,truth,targets)
        gains={u:{item:5-j for j,item in enumerate(truth[u])} for u in targets}
        labels=np.fromiter((gains[targets[r]].get(int(data['items'][c]),0) for r,c in zip(data['rr'],data['cc'])),dtype=np.int8)
        assert np.array_equal(binary>0,labels>0)
        xx.append(data['x'][valid]);yy.append(labels[valid]);gg.append(query_groups(data,valid))
        audit.append({'feature_cutoff':anchor,'label_end':end,'rows':int(valid.sum())})
    assert all(row['label_end']<=cutoff for row in audit)
    x=np.concatenate(xx);y=np.concatenate(yy);groups=np.concatenate(gg);del xx,yy,gg
    assert groups.sum()==len(y) and params['label_gain']==list(range(6))
    tick=time.perf_counter();model=lgb.LGBMRanker(**params).fit(x,y,group=groups)
    return model,None,{'outer_cutoff':cutoff,'inner_windows':audit,'outer_labels_used_in_fit':False,
      'training_rows':len(y),'query_count':len(groups),'group_sum_equals_rows':True,
      'label_rule':'Top5 membership gains 5..1 by inner future frequency rank; evaluation binary MAP@5 unchanged',
      'rank_fit_seconds':time.perf_counter()-tick}
