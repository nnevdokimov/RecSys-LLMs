"""Additional training examples from permitted past records, not other solutions."""
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sweettv.experiment import make_truth
from sweettv.nested_rerank import build_features,inner_windows,label_rows
from sweettv.models import row_max_norm,top_five


def select_extra_users(con,start,anchor,targets,cfg):
    excluded=set(targets)
    rows=con.execute(f"SELECT user_id,count(*) n FROM air_views WHERE air_start>='{start}' AND air_end<='{anchor}' AND fraction>=.8 GROUP BY 1 HAVING count(*)>={cfg['min_past_qualified_events']} ORDER BY n DESC,user_id").fetchall()
    return [u for u,n in rows if u not in excluded and (not cfg['reserve_non_template_user_id_mod5_zero'] or int(u)%5!=0)][:cfg['extra_users_per_inner_window']]


def fit_population(con,start,cutoff,targets,feature_cfg,cfg):
    base_x=[];base_y=[];extra_x=[];extra_y=[];audit=[]
    for anchor,end in inner_windows(start,cutoff,feature_cfg):
        data=build_features(con,start,anchor,end,targets,feature_cfg)
        truth,_=make_truth(con,anchor,end,targets);y,valid,_=label_rows(data,truth,targets)
        base_x.append(data['x'][valid]);base_y.append(y[valid])
        extra=select_extra_users(con,start,anchor,targets,cfg);positive=0;sample_rows=0
        for left in range(0,len(extra),cfg['batch_size']):
            cohort=extra[left:left+cfg['batch_size']]
            data=build_features(con,start,anchor,end,cohort,feature_cfg)
            assert np.all(data['history']>=cfg['min_past_qualified_events'])
            truth,_=make_truth(con,anchor,end,cohort);y,valid,_=label_rows(data,truth,cohort)
            extra_x.append(data['x'][valid]);extra_y.append(y[valid]);positive+=int(y[valid].sum());sample_rows+=int(valid.sum())
        audit.append({'feature_cutoff':anchor,'label_end':end,'extra_users':len(extra),'extra_candidate_rows':sample_rows,'extra_positive_rows':positive,'extra_user_ids':extra})
    assert all(row['label_end']<=cutoff for row in audit)
    bx=np.concatenate(base_x);by=np.concatenate(base_y);del base_x,base_y
    kwargs={**feature_cfg['estimator'],'random_state':feature_cfg['seed']}
    tick=time.perf_counter();reference=HistGradientBoostingClassifier(**kwargs).fit(bx,by);ref_seconds=time.perf_counter()-tick
    tick=time.perf_counter();augmented=HistGradientBoostingClassifier(**kwargs).fit(np.concatenate([bx]+extra_x),np.concatenate([by]+extra_y));aug_seconds=time.perf_counter()-tick
    return reference,augmented,{'outer_cutoff':cutoff,'inner_windows':audit,'outer_labels_used_in_fit':False,'template_fit_seconds':ref_seconds,'augmented_fit_seconds':aug_seconds}


def population_predictions(data,targets,reference,augmented):
    scores=[]
    for clf in [reference,augmented]:
        raw=np.zeros((len(targets),len(data['items'])))
        raw[data['rr'],data['cc']]=clf.predict_proba(data['x'])[:,1]
        scores.append(row_max_norm(raw))
    result={}
    for name,score in [('template_only_reference',scores[0]),('augmented_population',scores[1]),('population_fusion50',.5*scores[0]+.5*scores[1])]:
        score=score.copy();score[data['history']==0]=0
        result[name]={u:top_five(score[i],data['items'],data['pop']) for i,u in enumerate(targets)}
    return result
