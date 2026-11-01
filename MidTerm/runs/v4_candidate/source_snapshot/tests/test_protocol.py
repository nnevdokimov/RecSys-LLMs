import unittest
import datetime as dt

import numpy as np
import duckdb

from sweettv.metrics import ap_at_k, mean_ap
from sweettv.models import top_five, feature_dict, Models


class ProtocolTests(unittest.TestCase):
    def test_hand_ap(self):
        # R={10,20,30}; hits at ranks 1,3,5: (1+2/3+3/5)/3=34/45.
        self.assertAlmostEqual(ap_at_k([10,20,30],[10,99,20,88,30]),34/45)
        # Source slide 5: X=(A,B,D,C,E), Y=(D,A,E,B,C), R=(A,B,C).
        self.assertAlmostEqual(ap_at_k(['A','B','C'],['A','B','D','C','E']),11/12)
        self.assertAlmostEqual(ap_at_k(['A','B','C'],['D','A','E','B','C']),8/15)
        self.assertAlmostEqual(ap_at_k([10,20],[10,10,20]),(1+2/3)/2)
        self.assertEqual(ap_at_k([], [1,2,3,4,5]),0)
        self.assertEqual(ap_at_k([1,2],[8,9,7,6,5]),0)
        self.assertEqual(ap_at_k([1,2,3,4,5,6],[1,2,3,4,5]),1)

    def test_empty_user_policy(self):
        self.assertEqual(mean_ap({'a':[1],'b':[]},{'a':[1],'b':[1]}),1)
        self.assertEqual(mean_ap({'a':[1],'b':[]},{'a':[1],'b':[1]},exclude_empty=False),.5)

    def test_common_fallback(self):
        ids=np.array([1,2,3,4,5,6]);pop=np.array([5,4,3,2,1,0])
        self.assertEqual(top_five(np.zeros(6),ids,pop),[1,2,3,4,5])
        self.assertEqual(top_five(np.array([0,0,0,0,0,1]),ids,pop),[6,1,2,3,4])

    def test_half_open_overlap(self):
        c=duckdb.connect()
        # Broadcast [100,200), viewing [50,150): 50 seconds; touching [200,250) is 0.
        self.assertEqual(c.execute('SELECT greatest(0,least(150,200)-greatest(50,100))').fetchone()[0],50)
        self.assertEqual(c.execute('SELECT greatest(0,least(250,200)-greatest(200,100))').fetchone()[0],0)

    def test_nested_interval_union(self):
        c=duckdb.connect()
        q='''WITH t(id,s,e) AS (VALUES (1,0,100),(2,0,100),(3,10,20),(4,30,80),(5,90,120),(6,140,150)),
          p AS(SELECT *,max(e) OVER(ORDER BY s,e,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) pe FROM t),
          f AS(SELECT *,CASE WHEN pe IS NULL OR s>pe THEN 1 ELSE 0 END flag FROM p),
          g AS(SELECT *,sum(flag) OVER(ORDER BY s,e,id ROWS UNBOUNDED PRECEDING) island FROM f)
          SELECT sum(e-s) FROM(SELECT min(s) s,max(e) e FROM g GROUP BY island)'''
        self.assertEqual(c.execute(q).fetchone()[0],130)

    def test_no_feature_from_id(self):
        self.assertEqual(feature_dict({'item':12345}),{})

    def test_directional_association_and_no_self_rule(self):
        rows=[{'user_id':u,'item':i,'n':1} for u,items in [('a',[1,2]),('b',[1]),('c',[1]),('d',[2])] for i in items]
        m=Models(rows,{},['a','b','c','d'],{1,2,3,4,5},dt.datetime(2020,1,1),['b','d'])
        scores=m.association(shrink=0,min_pair=1)
        self.assertAlmostEqual(scores[0,1],1/3)  # P(item2 | item1)
        self.assertAlmostEqual(scores[1,0],1/2)  # P(item1 | item2)
        self.assertEqual(scores[0,0],0)
        self.assertEqual(scores[1,1],0)


if __name__=='__main__':
    unittest.main()
