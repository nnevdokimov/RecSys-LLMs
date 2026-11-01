"""CSV audit, bounded interval join and union of viewing fragments per airing."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import time
from pathlib import Path

import duckdb


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def records(con, sql):
    cur = con.execute(sql)
    return [dict(zip([x[0] for x in cur.description], row)) for row in cur.fetchall()]


def prepare(config):
    t0 = time.perf_counter()
    cfg = json.loads(Path(config).read_text())
    root = Path(cfg['data_dir'])
    Path('artifacts').mkdir(exist_ok=True)
    con = duckdb.connect('artifacts/data.duckdb')
    con.execute(f"SET threads={cfg['threads']}")
    con.execute(f"SET memory_limit='{cfg['memory_limit']}'")
    con.execute("SET temp_directory='tmp/duckdb'")
    audit = {'files': {}, 'schedule': [], 'timezone': 'No timezone offsets in either source; retain identical naive wall-clock times. Absolute timezone not established.'}
    for p in sorted(root.glob('*.csv')):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        audit['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
    schedule_files = sorted(root.glob('export*.csv'))
    paths = '[' + ','.join("'" + str(p).replace("'", "''") + "'" for p in schedule_files) + ']'
    con.execute(f"CREATE OR REPLACE TABLE schedule_raw AS SELECT * FROM read_csv({paths}, all_varchar=true, filename=true, strict_mode=true)")
    con.execute("""CREATE OR REPLACE TABLE schedule AS
      SELECT row_number() OVER (ORDER BY channel_id, s, e, item, tv_show_title) AS air_id, * FROM (
        SELECT DISTINCT try_cast(channel_id AS INTEGER) channel_id, channel_title,
        try_strptime(start_time, '%d.%m.%Y %H:%M:%S') s,
        try_strptime(start_time, '%d.%m.%Y %H:%M:%S') + try_cast(duration AS BIGINT)*INTERVAL '1 second' e,
        try_cast(duration AS BIGINT) duration, try_cast(tv_show_id AS BIGINT) item,
        tv_show_title, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
        year_of_production, director, actors
        FROM schedule_raw) WHERE channel_id IS NOT NULL AND s IS NOT NULL AND e>s AND item IS NOT NULL""")
    for p in schedule_files:
        q = "filename='"+str(p).replace("'", "''")+"'"
        entry = records(con, f"""SELECT count(*) rows, count(DISTINCT channel_id) channels,
          count(DISTINCT tv_show_id) FILTER(WHERE tv_show_id!='0') programs,
          count(*) FILTER(WHERE tv_show_id!='0') labelled_rows,
          min(try_strptime(start_time,'%d.%m.%Y %H:%M:%S')) start_min,
          max(try_strptime(start_time,'%d.%m.%Y %H:%M:%S')) start_max,
          count(*)-count(DISTINCT (channel_id,channel_title,start_time,duration,tv_show_title,tv_show_id,tv_show_category,tv_show_genre_1,tv_show_genre_2,tv_show_genre_3,year_of_production,director,actors)) exact_duplicate_rows,
          count(*) FILTER(WHERE try_strptime(start_time,'%d.%m.%Y %H:%M:%S') IS NULL OR try_cast(duration AS BIGINT)<=0 OR try_cast(duration AS BIGINT) IS NULL) invalid_time_or_duration,
          quantile_cont(try_cast(duration AS BIGINT),[0,0.25,0.5,0.75,0.95,0.99,1]) duration_quantiles
          FROM schedule_raw WHERE {q}""")[0]
        entry['file'] = p.name
        entry['nulls'] = records(con, f"SELECT " + ','.join(f'count(*) FILTER(WHERE "{c}" IS NULL) AS "{c}"' for c in ['channel_id','channel_title','start_time','duration','tv_show_title','tv_show_id','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','year_of_production','director','actors']) + f' FROM schedule_raw WHERE {q}')[0]
        audit['schedule'].append(entry)
    con.execute(f"CREATE OR REPLACE TABLE logs_raw AS SELECT * FROM read_csv('{root}/dataset11-30.csv', all_varchar=true, strict_mode=true)")
    audit['logs'] = records(con, """SELECT count(*) rows, count(DISTINCT user_id) users, count(DISTINCT vsetv_id) channels,
       min(start_time) start_min, max(start_time) start_max, max(stop_time) end_max,
       count(*)-count(DISTINCT (user_id,vsetv_id,start_time,stop_time,duraton)) exact_duplicates,
       count(*) FILTER(WHERE try_cast(start_time AS TIMESTAMP) IS NULL OR try_cast(stop_time AS TIMESTAMP) IS NULL OR try_cast(stop_time AS TIMESTAMP)<=try_cast(start_time AS TIMESTAMP)) invalid_times,
       count(*) FILTER(WHERE try_cast(duraton AS BIGINT) != epoch(try_cast(stop_time AS TIMESTAMP)-try_cast(start_time AS TIMESTAMP))) duration_disagreements,
       quantile_cont(try_cast(duraton AS BIGINT),[0,0.25,0.5,0.75,0.95,0.99,1]) duration_quantiles
       FROM logs_raw""")[0]
    audit['logs']['nulls'] = records(con, 'SELECT '+','.join(f'count(*) FILTER(WHERE "{c}" IS NULL) AS "{c}"' for c in ['user_id','vsetv_id','start_time','stop_time','duraton'])+' FROM logs_raw')[0]
    con.execute("""CREATE OR REPLACE TABLE logs AS SELECT row_number() OVER() log_id,* FROM (
      SELECT DISTINCT user_id, try_cast(vsetv_id AS INTEGER) channel_id,
      try_cast(start_time AS TIMESTAMP) s, try_cast(stop_time AS TIMESTAMP) e FROM logs_raw)
      WHERE user_id IS NOT NULL AND channel_id IS NOT NULL AND s IS NOT NULL AND e>s""")
    audit['channel_mapping'] = records(con,"""SELECT count(*) logs, count(*) FILTER(WHERE channel_id IN (SELECT channel_id FROM schedule)) matching_channel_logs,
      count(DISTINCT channel_id) FILTER(WHERE channel_id NOT IN(SELECT channel_id FROM schedule)) missing_channels FROM logs""")[0]
    audit['missing_channels'] = records(con,"SELECT channel_id,count(*) logs FROM logs WHERE channel_id NOT IN(SELECT channel_id FROM schedule) GROUP BY 1 ORDER BY 2 DESC")
    con.execute(f"CREATE OR REPLACE TABLE template AS SELECT * FROM read_csv('{root}/submission.csv', all_varchar=true)")
    audit['template'] = records(con,"SELECT count(*) rows, count(DISTINCT user_id) users, count(*) FILTER(WHERE regexp_full_match(tv_show_id,'[0-9]+( [0-9]+){4}')) valid_five_lists FROM template")[0]
    audit['template']['columns'] = ['user_id','tv_show_id']
    audit['template']['users_absent_from_logs'] = con.execute("SELECT count(*) FROM template WHERE user_id NOT IN (SELECT user_id FROM logs)").fetchone()[0]
    print('Raw audit complete', flush=True)
    dump('artifacts/data_audit.json', audit)
    # Day bucketing bounds candidates before exact half-open interval comparisons.
    con.execute("""CREATE OR REPLACE TABLE schedule_days AS
       SELECT a.*, d.day FROM schedule a, LATERAL generate_series(date_trunc('day',s),date_trunc('day',e-INTERVAL '1 microsecond'),INTERVAL '1 day') d(day)""")
    con.execute("""CREATE OR REPLACE TABLE log_days AS
       SELECT l.*, d.day FROM logs l, LATERAL generate_series(date_trunc('day',s),date_trunc('day',e-INTERVAL '1 microsecond'),INTERVAL '1 day') d(day)""")
    audit['coverage_any_airtime'] = records(con,"""SELECT count(DISTINCT l.log_id) matched_logs FROM log_days l JOIN schedule_days a
       ON l.channel_id=a.channel_id AND l.day=a.day AND l.s<a.e AND l.e>a.s""")[0]
    con.execute("""CREATE OR REPLACE TABLE view_segments AS SELECT DISTINCT l.log_id,l.user_id,a.air_id,a.item,a.channel_id,a.s air_start,a.e air_end,a.duration,
       greatest(l.s,a.s) s, least(l.e,a.e) e FROM log_days l JOIN schedule_days a
       ON l.channel_id=a.channel_id AND l.day=a.day AND l.s<a.e AND l.e>a.s WHERE a.item>0""")
    audit['labelled_overlap'] = records(con,"""SELECT count(*) interval_pairs,count(DISTINCT log_id) matched_logs,count(DISTINCT user_id) users,
       quantile_cont(epoch(e-s),[0,0.25,0.5,0.75,0.95,0.99,1]) overlap_seconds_quantiles,
       quantile_cont(epoch(e-s)/duration,[0,0.25,0.5,0.75,0.95,0.99,1]) overlap_fraction_quantiles FROM view_segments""")[0]
    audit['labelled_overlap']['logs_crossing_multiple_airings'] = con.execute('SELECT count(*) FROM (SELECT log_id FROM view_segments GROUP BY 1 HAVING count(*)>1)').fetchone()[0]
    # Running MAX, rather than LAG(e), is required for nested intervals.
    con.execute("""CREATE OR REPLACE TABLE air_views AS WITH prev AS (
       SELECT *,max(e) OVER(PARTITION BY user_id,air_id ORDER BY s,e,log_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end FROM view_segments),
       flags AS (SELECT *,CASE WHEN prev_end IS NULL OR s>prev_end THEN 1 ELSE 0 END new_island FROM prev),
       islands AS (SELECT *,sum(new_island) OVER(PARTITION BY user_id,air_id ORDER BY s,e,log_id ROWS UNBOUNDED PRECEDING) island FROM flags),
       merged AS (SELECT user_id,air_id,item,channel_id,air_start,air_end,duration,island,min(s) s,max(e) e FROM islands GROUP BY ALL)
       SELECT user_id,air_id,item,channel_id,air_start,air_end,duration,sum(epoch(e-s)) watched_seconds,
       sum(epoch(e-s))/duration fraction FROM merged
       GROUP BY user_id,air_id,item,channel_id,air_start,air_end,duration""")
    audit['air_views'] = records(con,"""SELECT count(*) user_air_pairs,count(*) FILTER(WHERE fraction>=0.8) fraction_80_pairs,
       count(*) FILTER(WHERE watched_seconds>=300) absolute_300_pairs, max(fraction) max_fraction,
       quantile_cont(fraction,[0,0.25,0.5,0.75,0.95,0.99,1]) fraction_quantiles FROM air_views""")[0]
    assert audit['air_views']['max_fraction'] <= 1.00000001, 'Union exceeded full broadcast duration'
    audit['schedule_conflicts'] = records(con,"""SELECT count(*) conflicting_slots FROM (
      SELECT channel_id,s FROM schedule GROUP BY 1,2 HAVING count(*)>1)""")[0]
    audit['preparation_seconds'] = time.perf_counter()-t0
    audit['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump('artifacts/data_audit.json',audit)
    con.execute("COPY air_views TO 'artifacts/air_views.parquet' (FORMAT PARQUET)")
    dump('artifacts/environment.json',{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'libraries':{n:importlib.metadata.version(n) for n in ['duckdb','numpy','scipy','scikit-learn','reportlab','pypdf']}})
    print(json.dumps(audit,ensure_ascii=False,default=str,indent=2),flush=True)
    con.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.json')
    prepare(parser.parse_args().config)
