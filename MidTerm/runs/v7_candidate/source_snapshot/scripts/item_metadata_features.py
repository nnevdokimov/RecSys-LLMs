"""Past categorical identities plus future ID/channel/time-derived durations."""
from collections import Counter
import numpy as np

FIELDS=['tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3']


def past_metadata(con,cutoff):
    result={}
    for field in FIELDS:
        query=f'''SELECT item,{field} FROM (SELECT item,{field},row_number() OVER(PARTITION BY item ORDER BY count(*) DESC,{field}) rn
          FROM schedule WHERE item>0 AND s<'{cutoff}' AND {field} IS NOT NULL GROUP BY 1,2) WHERE rn=1'''
        for item,value in con.execute(query).fetchall():result.setdefault(int(item),{})[field]=value
    return result


def fit_vocabulary(con,cutoff,top_genres=20):
    meta=past_metadata(con,cutoff)
    categories=sorted({row['tv_show_category'] for row in meta.values() if row.get('tv_show_category')})
    counts=Counter(v for row in meta.values() for field,v in row.items() if field.startswith('tv_show_genre') and v)
    genres=[v for v,n in sorted(counts.items(),key=lambda p:(-p[1],p[0]))[:top_genres]]
    return {'fitted_until_exclusive':cutoff,'categories':categories,'genres':genres}


def augment_item_metadata(data,con,cutoff,end,vocabulary):
    meta=past_metadata(con,cutoff)
    names=['meta_category='+x for x in vocabulary['categories']]+['meta_genre='+x for x in vocabulary['genres']]
    names+=['meta_log_future_mean_duration','meta_log_future_std_duration','meta_log_future_channels']
    matrix=np.zeros((len(data['items']),len(names)),np.float32)
    for j,item in enumerate(data['items']):
        row=meta.get(int(item),{});genres={row.get(f) for f in FIELDS[1:]}
        matrix[j,:len(vocabulary['categories'])]=[row.get('tv_show_category')==c for c in vocabulary['categories']]
        matrix[j,len(vocabulary['categories']):len(vocabulary['categories'])+len(vocabulary['genres'])]=[g in genres for g in vocabulary['genres']]
    # These are exclusively schedule availability/timing facts, no future labels.
    ix={int(item):j for j,item in enumerate(data['items'])}
    for item,mean,std,nch in con.execute(f'''SELECT item,avg(duration),stddev_pop(duration),count(DISTINCT channel_id)
      FROM schedule WHERE item>0 AND s>='{cutoff}' AND e<='{end}' GROUP BY 1''').fetchall():
        if int(item) in ix:matrix[ix[int(item)],-3:]=np.log1p([mean,std or 0,nch])
    extra=matrix[data['cc']]
    return {**data,'x':np.column_stack([data['x'],extra]).astype(np.float32),'feature_names':np.concatenate([data['feature_names'],np.array(names)])}
