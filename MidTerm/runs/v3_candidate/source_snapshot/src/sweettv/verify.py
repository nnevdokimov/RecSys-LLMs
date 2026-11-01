"""Independent interval-union cross-check and data/protocol invariants."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import duckdb

from .prepare import dump


def main():
    c=duckdb.connect('artifacts/data.duckdb',read_only=True)
    chosen=c.execute('''WITH sampled AS (SELECT * FROM air_views ORDER BY hash(user_id,air_id) LIMIT 200)
      SELECT a.user_id,a.air_id,a.duration,a.watched_seconds,v.s,v.e
      FROM sampled a JOIN view_segments v ON a.user_id=v.user_id AND a.air_id=v.air_id
      ORDER BY a.user_id,a.air_id,v.s,v.e''').fetchall()
    groups=defaultdict(list);expected={}
    for u,a,d,w,s,e in chosen:
        groups[u,a].append((s,e));expected[u,a]=w
    for key,intervals in groups.items():
        merged=[]
        for s,e in intervals:
            if not merged or s>merged[-1][1]:merged.append([s,e])
            else:merged[-1][1]=max(merged[-1][1],e)
        actual=sum((e-s).total_seconds() for s,e in merged)
        assert actual==expected[key],(key,actual,expected[key])
    fixed=c.execute("SELECT fraction FROM air_views WHERE user_id='1388513685702887520' AND air_id=11152").fetchone()[0]
    assert fixed==1
    assert c.execute('SELECT count(*) FROM air_views WHERE fraction>1 OR fraction<=0').fetchone()[0]==0
    extras={
      'sampled_unions_checked_independently':len(groups),
      'python_union_equals_sql_for_all_sampled':True,
      'previous_duplicate_clip_failure_fraction_after_fix':fixed,
      'all_fractions_in_0_1':True,
      'logical_cpu_count':__import__('os').cpu_count(),
      'users_without_any_80pct_history':c.execute('SELECT count(*) FROM (SELECT DISTINCT user_id FROM logs EXCEPT SELECT DISTINCT user_id FROM air_views WHERE fraction>=0.8)').fetchone()[0],
      'schedule_rows_overlapping_prior_airing_on_same_channel':c.execute('SELECT count(*) FROM(SELECT s,max(e) OVER(PARTITION BY channel_id ORDER BY s ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) pe FROM schedule) WHERE s<pe').fetchone()[0],
      'logs_with_80pct_matched_airing':c.execute('SELECT count(DISTINCT v.log_id) FROM view_segments v JOIN air_views a ON v.user_id=a.user_id AND v.air_id=a.air_id WHERE a.fraction>=0.8').fetchone()[0]
    }
    # A future-only attribute must not enter the training metadata vocabulary.
    from .experiment import load_metadata
    q=duckdb.connect()
    q.execute('CREATE TABLE schedule AS SELECT * FROM read_parquet(\'artifacts/air_views.parquet\') WHERE false')
    q.execute('DROP TABLE schedule')
    q.execute('''CREATE TABLE schedule(item BIGINT,s TIMESTAMP,tv_show_title VARCHAR,tv_show_category VARCHAR,
      tv_show_genre_1 VARCHAR,tv_show_genre_2 VARCHAR,tv_show_genre_3 VARCHAR,year_of_production VARCHAR,director VARCHAR,actors VARCHAR)''')
    q.execute("INSERT INTO schedule(item,s,tv_show_title) VALUES(1,'2020-01-01','known'),(1,'2020-03-01','future'),(2,'2020-03-01','future-only')")
    meta=load_metadata(q,'2020-02-01')
    assert meta=={1:{'tv_show_title':'known'}},meta
    extras['future_metadata_excluded_by_real_loader_test']=True
    dump('artifacts/integration_checks.json',extras)
    print(json.dumps(extras,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
