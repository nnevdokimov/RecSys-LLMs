"""Query grouping and monotone ranking-score conversion for LightGBM."""
import numpy as np
from sweettv.models import top_five,row_max_norm


def query_groups(data,valid):
    rr=data['rr'][valid]
    assert len(rr)>0 and np.all(rr[1:]>=rr[:-1])
    counts=np.bincount(rr);groups=counts[counts>0].astype(np.int32)
    assert groups.sum()==len(rr)
    return groups


def vector_to_scores(data,targets,vector,raw_rank_score=False):
    assert np.isfinite(vector).all()
    if raw_rank_score:
        raw=np.full((len(targets),len(data['items'])),-np.inf)
        raw[data['rr'],data['cc']]=vector
        # Rank margins are not probabilities. Exp is a monotone positive map,
        # preserving each query's order and leaving ungenerated items at zero.
        result=np.exp(raw-np.max(raw,axis=1,keepdims=True))
    else:
        result=np.zeros((len(targets),len(data['items'])))
        result[data['rr'],data['cc']]=vector;result=row_max_norm(result)
    result[data['history']==0]=0
    return result


def ranked_predictions(data,targets,score):
    return {u:top_five(score[i],data['items'],data['pop']) for i,u in enumerate(targets)}
