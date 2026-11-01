"""Dense count-share labels INSIDE the allowed inner forecast window."""
import numpy as np


def frequency_share_targets(data,con,start,end,targets):
    rows=con.execute(f'''SELECT user_id,item,count(*) n,sum(count(*)) OVER(PARTITION BY user_id) total
      FROM air_views WHERE fraction>=.8 AND air_start>='{start}' AND air_end<='{end}' AND item>0 GROUP BY 1,2''').fetchall()
    lookup={(u,int(item)):100*n/total for u,item,n,total in rows}
    return np.array([lookup.get((targets[r],int(data['items'][c])),0) for r,c in zip(data['rr'],data['cc'])],dtype=np.float32)
