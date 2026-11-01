"""Shared causal feature/fit path for the selected 86-feature ranker."""
import time
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sweettv.nested_rerank import build_features,inner_windows,label_rows
from sweettv.experiment import make_truth
from habit_features import augment_habit
from item_metadata_features import augment_item_metadata
from lambdarank_helpers import query_groups,vector_to_scores,ranked_predictions


def feature_data(con,start,cutoff,end,targets,feature_cfg,vocabulary):
    base=build_features(con,start,cutoff,end,targets,feature_cfg)
    data=augment_item_metadata(augment_habit(base,con,start,cutoff,targets),con,cutoff,end,vocabulary)
    assert data['x'].shape[1]==86
    return data


def fit_ranker(con,start,cutoff,targets,feature_cfg,vocabulary,params,include_reference=False):
    xx=[];yy=[];gg=[];audit=[]
    for anchor,end in inner_windows(start,cutoff,feature_cfg):
        data=feature_data(con,start,anchor,end,targets,feature_cfg,vocabulary)
        truth,_=make_truth(con,anchor,end,targets);y,valid,_=label_rows(data,truth,targets)
        xx.append(data['x'][valid]);yy.append(y[valid]);gg.append(query_groups(data,valid))
        audit.append({'feature_cutoff':anchor,'label_end':end,'rows':int(valid.sum())})
    assert all(row['label_end']<=cutoff for row in audit)
    x=np.concatenate(xx);y=np.concatenate(yy);groups=np.concatenate(gg);del xx,yy,gg
    assert groups.sum()==len(y)
    tick=time.perf_counter();ranker=lgb.LGBMRanker(**params).fit(x,y,group=groups);rank_seconds=time.perf_counter()-tick
    classifier=None
    if include_reference:
        classifier=HistGradientBoostingClassifier(**feature_cfg['estimator'],random_state=feature_cfg['seed']).fit(x[:,:45],y)
    return ranker,classifier,{'outer_cutoff':cutoff,'inner_windows':audit,'outer_labels_used_in_fit':False,'training_rows':len(y),'query_count':len(groups),'group_sum_equals_rows':True,'rank_fit_seconds':rank_seconds}


def predict_ranker(data,targets,ranker):
    return ranked_predictions(data,targets,vector_to_scores(data,targets,ranker.booster_.predict(data['x']),True))


def predict_reference(data,targets,classifier):
    return ranked_predictions(data,targets,vector_to_scores(data,targets,classifier.predict_proba(data['x'][:,:45])[:,1]))
