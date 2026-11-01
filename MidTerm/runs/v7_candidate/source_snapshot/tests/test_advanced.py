import datetime as dt
import unittest
import numpy as np
import duckdb

from sweettv.advanced import als_scores
from sweettv.models import Models,row_max_norm
from sweettv.context_experiment import time_context
from sweettv.nested_rerank import inner_windows
from sweettv.category_context import category_time_features
from sweettv.partial_views import partial_view_data,selected_partial_features
from scripts.habit_features import augment_habit
from scripts.item_metadata_features import augment_item_metadata,fit_vocabulary
from scripts.frequency_targets import frequency_share_targets
from scripts.lambdarank_helpers import query_groups,vector_to_scores
from lightgbm import LGBMRanker


class AdvancedTests(unittest.TestCase):
    def test_lambdarank_grouping_order_and_determinism(self):
        x=np.array([[2.,0.],[1.,0.],[2.,1.],[1.,1.]])
        y=np.array([1,0,1,0])
        groups=query_groups({'rr':np.array([0,0,1,1])},np.ones(4,bool))
        np.testing.assert_array_equal(groups,[2,2])
        params={'objective':'lambdarank','n_estimators':20,'num_leaves':3,'min_child_samples':1,'min_data_in_bin':1,'reg_lambda':1,'random_state':42,'n_jobs':1,'verbosity':-1,'deterministic':True,'force_col_wise':True}
        a=LGBMRanker(**params).fit(x,y,group=groups).booster_.predict(x)
        b=LGBMRanker(**params).fit(x,y,group=groups).booster_.predict(x)
        np.testing.assert_array_equal(a,b)
        self.assertGreater(a[0],a[1]);self.assertGreater(a[2],a[3])
        data={'rr':np.array([0,0,1,1]),'cc':np.array([0,1,0,1]),'items':np.arange(5),'history':np.array([1,0])}
        scores=vector_to_scores(data,['a','cold'],a,True)
        self.assertGreater(scores[0,0],scores[0,1]);self.assertEqual(scores[0,2],0)
        np.testing.assert_array_equal(scores[1],np.zeros(5))

    def test_frequency_target_uses_all_counts_inside_inner_label_window(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE air_views(user_id VARCHAR,item BIGINT,air_start TIMESTAMP,air_end TIMESTAMP,fraction DOUBLE)')
        c.execute("INSERT INTO air_views VALUES('a',1,'2020-01-14 10:00','2020-01-14 11:00',1),('a',1,'2020-01-15 10:00','2020-01-15 11:00',1),('a',2,'2020-01-16 10:00','2020-01-16 11:00',1)")
        data={'rr':np.array([0,0,0]),'cc':np.array([0,1,2]),'items':np.array([1,2,3])}
        before=frequency_share_targets(data,c,'2020-01-13','2020-01-20',['a'])
        np.testing.assert_allclose(before,[200/3,100/3,0],atol=1e-5)
        c.execute("INSERT INTO air_views VALUES('a',3,'2020-01-21 10:00','2020-01-21 11:00',1)")
        after=frequency_share_targets(data,c,'2020-01-13','2020-01-20',['a'])
        np.testing.assert_array_equal(before,after)

    def test_item_metadata_uses_past_labels_and_future_timing_only(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE schedule(item BIGINT,channel_id INTEGER,s TIMESTAMP,e TIMESTAMP,duration INTEGER,tv_show_category VARCHAR,tv_show_genre_1 VARCHAR,tv_show_genre_2 VARCHAR,tv_show_genre_3 VARCHAR)')
        c.execute("INSERT INTO schedule VALUES(1,1,'2020-01-06 10:00','2020-01-06 11:00',3600,'Kids','Cartoon',NULL,NULL),(1,1,'2020-01-13 10:00','2020-01-13 11:00',3600,'Contradictory future','News',NULL,NULL),(2,2,'2020-01-13 10:00','2020-01-13 11:00',3600,'Kids','Cartoon',NULL,NULL)")
        vocab=fit_vocabulary(c,'2020-01-13',20)
        self.assertEqual(vocab['categories'],['Kids'])
        data={'rr':np.array([0,0]),'cc':np.array([0,1]),'items':np.array([1,2]),'x':np.zeros((2,45),np.float32),'feature_names':np.array([f'f{i}' for i in range(45)])}
        before=augment_item_metadata(data,c,'2020-01-13','2020-01-20',vocab)
        self.assertEqual(before['x'][0,45],1)
        self.assertEqual(before['x'][1,45],0) # future-only category is unknown
        self.assertGreater(before['x'][1,-3],0) # its allowed future duration is usable
        c.execute("UPDATE schedule SET tv_show_category='Altered future',tv_show_genre_1='Altered genre' WHERE s>='2020-01-13'")
        after=augment_item_metadata(data,c,'2020-01-13','2020-01-20',vocab)
        np.testing.assert_array_equal(before['x'],after['x'])

    def test_habit_features_normalize_supply_and_exclude_future(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE air_views(user_id VARCHAR,item BIGINT,air_start TIMESTAMP,air_end TIMESTAMP,fraction DOUBLE)')
        c.execute("INSERT INTO air_views VALUES('a',1,'2020-01-06 10:00','2020-01-06 11:00',1),('a',1,'2020-01-13 10:00','2020-01-13 11:00',1),('a',1,'2020-01-27 10:00','2020-01-27 11:00',1)")
        c.execute('CREATE TABLE schedule(item BIGINT,s TIMESTAMP,e TIMESTAMP)')
        c.execute("INSERT INTO schedule VALUES(1,'2020-01-06 10:00','2020-01-06 11:00'),(1,'2020-01-13 10:00','2020-01-13 11:00'),(1,'2020-01-27 10:00','2020-01-27 11:00'),(1,'2020-01-28 10:00','2020-01-28 11:00')")
        data={'rr':np.array([0,0]),'cc':np.array([0,1]),'items':np.array([1,2]),'x':np.zeros((2,45),np.float32),'feature_names':np.array([f'f{i}' for i in range(45)])}
        before=augment_habit(data,c,'2020-01-06','2020-02-03',['a'])
        self.assertEqual(before['x'].shape,(2,56))
        self.assertAlmostEqual(before['x'][0,47],.75)
        self.assertAlmostEqual(before['x'][0,50],.5)
        self.assertAlmostEqual(before['x'][0,51],1)
        self.assertLess(before['x'][0,52],0)
        c.execute("INSERT INTO air_views VALUES('a',1,'2020-02-04 10:00','2020-02-04 11:00',1)")
        c.execute("INSERT INTO schedule VALUES(1,'2020-02-04 10:00','2020-02-04 11:00')")
        after=augment_habit(data,c,'2020-01-06','2020-02-03',['a'])
        np.testing.assert_array_equal(before['x'],after['x'])

    def test_partial_views_are_continuous_and_causal(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE air_views(user_id VARCHAR,item BIGINT,air_start TIMESTAMP,air_end TIMESTAMP,fraction DOUBLE,watched_seconds DOUBLE)')
        c.execute("INSERT INTO air_views VALUES('a',1,'2020-01-06 10:00','2020-01-06 11:00',.8,2880),('a',1,'2020-01-07 10:00','2020-01-07 11:00',.4,1440)")
        m=Models([{'user_id':'a','item':1,'n':1}],{},['a','cold'],{1,2,3,4,5},dt.datetime(2020,1,13))
        score,lookup=partial_view_data(c,m,'2020-01-06','2020-01-13')
        self.assertAlmostEqual(score[0,0],1.2)
        features=selected_partial_features(lookup,[0,0],[0,1])
        self.assertAlmostEqual(features['partial_mean_fraction'][0],.6,places=6)
        self.assertEqual(features['partial_qualified_share'][0],.5)
        self.assertEqual(features['partial_log_any_count'][1],0)
        c.execute("INSERT INTO air_views VALUES('a',2,'2020-01-14 10:00','2020-01-14 11:00',1,3600)")
        later,again=partial_view_data(c,m,'2020-01-06','2020-01-13')
        np.testing.assert_array_equal(score,later);self.assertEqual(lookup,again)

    def test_category_clock_uses_only_past_categories_and_views(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE schedule(item BIGINT,s TIMESTAMP,e TIMESTAMP,duration INTEGER,tv_show_category VARCHAR)')
        c.execute("INSERT INTO schedule VALUES(1,'2020-01-13 10:00','2020-01-13 11:00',3600,'Future contradictory label'),(2,'2020-01-13 20:00','2020-01-13 21:00',3600,'News'),(3,'2020-01-13 10:00','2020-01-13 11:00',3600,'Kids'),(4,'2020-01-13 20:00','2020-01-13 21:00',3600,'Kids')")
        c.execute('CREATE TABLE air_views(user_id VARCHAR,item BIGINT,air_start TIMESTAMP,air_end TIMESTAMP,duration INTEGER,fraction DOUBLE)')
        c.execute("INSERT INTO air_views VALUES('a',1,'2020-01-06 10:00','2020-01-06 11:00',3600,1),('a',2,'2020-01-06 20:00','2020-01-06 21:00',3600,1)")
        m=Models([{'user_id':'a','item':1,'n':1},{'user_id':'a','item':2,'n':1}],{1:{'tv_show_category':'Kids'},2:{'tv_show_category':'News'},4:{'tv_show_category':'Kids'}},['a','cold'],{1,2,3,4,5},dt.datetime(2020,1,13))
        before=category_time_features(c,m,'2020-01-06','2020-01-13','2020-01-20',1,.1,28)
        self.assertGreater(before['category_clock_affinity'][0,0],before['category_clock_affinity'][0,3])
        self.assertEqual(before['category_known'][0,2],0) # future-only Kids label is not consumed
        self.assertEqual(before['category_clock_affinity'][0,2],0)
        np.testing.assert_array_equal(before['category_clock_affinity'][1],np.zeros(5))
        c.execute("UPDATE schedule SET tv_show_category='Changed future category'")
        c.execute("INSERT INTO air_views VALUES('a',4,'2020-01-14 20:00','2020-01-14 21:00',3600,1)")
        after=category_time_features(c,m,'2020-01-06','2020-01-13','2020-01-20',1,.1,28)
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_normalization_accepts_integer_popularity(self):
        np.testing.assert_array_equal(row_max_norm(np.array([[1,2],[0,0]])),[[.5,1],[0,0]])

    def test_nested_training_labels_end_before_outer_holdout(self):
        cfg={'inner_min_train_days':14,'inner_step_days':14,'inner_max_forecast_days':56,'inner_min_forecast_days':14}
        windows=inner_windows('2020-03-09','2020-04-20',cfg)
        self.assertEqual(windows,[('2020-03-23 00:00:00','2020-04-20 00:00:00'),('2020-04-06 00:00:00','2020-04-20 00:00:00')])
        self.assertTrue(all(end<='2020-06-15 00:00:00' for _,end in inner_windows('2020-03-09','2020-06-15',cfg)))

    def test_als_minimizes_actual_weighted_objective(self):
        rows=[{'user_id':u,'item':i,'n':n} for u,i,n in [('a',1,10),('a',2,3),('b',2,6),('b',3,2)]]
        m=Models(rows,{},['a','b','cold'],{1,2,3,4,5},dt.datetime(2020,1,1))
        fitted=als_scores(m,factors=3,alpha=10,reg=1,iterations=8,seed=42,return_diagnostics=True)
        trace=fitted['objective_trace']
        self.assertTrue(all(b<=a+1e-8 for a,b in zip(trace,trace[1:])),trace)
        self.assertLess(trace[-1],trace[0]/2)
        self.assertTrue(np.isfinite(fitted['scores']).all())
        np.testing.assert_array_equal(fitted['scores'][2],np.zeros(5))
        again=als_scores(m,factors=3,alpha=10,reg=1,iterations=8,seed=42)
        np.testing.assert_allclose(again,fitted['scores'],atol=1e-12)

    def test_clock_context_is_causal_and_integrates_airing_duration(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE schedule(channel_id INTEGER,item BIGINT,s TIMESTAMP,e TIMESTAMP,duration INTEGER)')
        c.execute("INSERT INTO schedule VALUES(1,1,'2020-01-06 08:00','2020-01-06 09:00',3600),(1,2,'2020-01-13 08:00','2020-01-13 16:00',28800),(2,3,'2020-01-13 08:00','2020-01-13 09:00',3600)")
        c.execute('CREATE TABLE logs(user_id VARCHAR,channel_id INTEGER,s TIMESTAMP,e TIMESTAMP)')
        c.execute("INSERT INTO logs VALUES('a',1,'2020-01-06 08:00','2020-01-06 12:00'),('a',2,'2020-01-14 08:00','2020-01-14 12:00')")
        model=Models([{'user_id':'a','item':1,'n':1}],{},['a','cold'],{1,2,3,4,5},dt.datetime(2020,1,13))
        f,p=time_context(c,model,'2020-01-06','2020-01-13','2020-01-20',0,'mean')
        self.assertAlmostEqual(p[0,model.ix[1]],1)  # calibration to the one training event
        self.assertAlmostEqual(f[0,model.ix[2]],.5) # 4/8 hours in the observed time block
        self.assertEqual(f[0,model.ix[3]],0)        # future channel-2 view must not leak
        np.testing.assert_array_equal(f[1],np.zeros(5))
        # Finer blocks preserve this four-hour viewing interval; day-specific
        # features keep the same Monday context. Future views remain excluded.
        for hours,days in [(2,'weekend'),(1,'weekend'),(4,'weekday'),(2,'weekday')]:
            finer,_=time_context(c,model,'2020-01-06','2020-01-13','2020-01-20',0,'mean',hours,days)
            np.testing.assert_allclose(finer,f,atol=1e-12)

    def test_soft_clock_wraps_midnight_and_trend_ignores_future(self):
        c=duckdb.connect()
        c.execute('CREATE TABLE schedule(channel_id INTEGER,item BIGINT,s TIMESTAMP,e TIMESTAMP,duration INTEGER)')
        c.execute("INSERT INTO schedule VALUES(1,1,'2020-01-06 23:00','2020-01-07 00:00',3600),(1,2,'2020-01-13 00:00','2020-01-13 01:00',3600),(1,3,'2020-01-13 22:00','2020-01-13 23:00',3600),(1,4,'2020-01-13 12:00','2020-01-13 13:00',3600),(2,5,'2020-01-13 12:00','2020-01-13 13:00',3600)")
        c.execute('CREATE TABLE logs(user_id VARCHAR,channel_id INTEGER,s TIMESTAMP,e TIMESTAMP)')
        c.execute("INSERT INTO logs VALUES('a',1,'2020-01-06 23:00','2020-01-07 00:00')")
        rows=[{'user_id':'a','item':1,'n':1,'n28':2**(-6/28)}]
        m=Models(rows,{},['a','cold'],{1,2,3,4,5},dt.datetime(2020,1,13))
        f,_=time_context(c,m,'2020-01-06','2020-01-13','2020-01-20',0,'mean',1,'weekend',3)
        self.assertAlmostEqual(f[0,m.ix[2]],f[0,m.ix[3]])
        self.assertGreater(f[0,m.ix[2]],f[0,m.ix[4]])
        before,_=time_context(c,m,'2020-01-06','2020-01-13','2020-01-20',7,'mean',1,'weekend',3,28)
        c.execute("INSERT INTO logs VALUES('a',2,'2020-01-15 00:00','2020-01-20 00:00')")
        after,_=time_context(c,m,'2020-01-06','2020-01-13','2020-01-20',7,'mean',1,'weekend',3,28)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(after[0,m.ix[5]],0)
        np.testing.assert_array_equal(after[1],np.zeros(5))


if __name__=='__main__':unittest.main()
