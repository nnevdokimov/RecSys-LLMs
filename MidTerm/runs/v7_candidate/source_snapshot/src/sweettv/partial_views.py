"""Past partial programme views as continuous evidence, never dislikes/labels."""
import datetime as dt
import numpy as np


FEATURE_NAMES=['partial_log_any_count','partial_log_half_count','partial_mean_fraction',
               'partial_qualified_share','partial_log_seconds','partial_recent_fraction_share','partial_last_recency']


def partial_view_data(con,model,start,cutoff):
    tix={u:i for i,u in enumerate(model.target_users)};cix={int(i):j for j,i in enumerate(model.candidates)}
    score=np.zeros((len(tix),len(cix)));lookup={};cut=dt.datetime.fromisoformat(cutoff)
    rows=con.execute(f'''SELECT user_id,item,count(*) n,count(*) FILTER(WHERE fraction>=.5) half_n,
        sum(fraction) fraction_sum,avg(fraction) mean_fraction,
        count(*) FILTER(WHERE fraction>=.8)*1.0/count(*) qualified_share,
        sum(watched_seconds) watched_total,
        sum(fraction*pow(2.0,-epoch(TIMESTAMP '{cutoff}'-air_end)/86400/28)) recent_fraction,
        max(air_end) last_time
      FROM air_views WHERE air_start>='{start}' AND air_end<='{cutoff}' AND fraction>0
      GROUP BY 1,2''').fetchall()
    for user,item,n,half_n,total,mean,qualified,seconds,recent,last in rows:
        if user in tix and item in cix:
            key=(tix[user],cix[item]);score[key]=total
            lookup[key]=(np.log1p(n),np.log1p(half_n),mean,float(qualified),np.log1p(seconds),
                         recent/total,2**(-max(0,(cut-last).total_seconds()/86400)/28))
    return score,lookup


def selected_partial_features(lookup,rr,cc):
    matrix=np.array([lookup.get((int(r),int(c)),(0,)*len(FEATURE_NAMES)) for r,c in zip(rr,cc)],dtype=np.float32)
    return {name:matrix[:,i] for i,name in enumerate(FEATURE_NAMES)}
