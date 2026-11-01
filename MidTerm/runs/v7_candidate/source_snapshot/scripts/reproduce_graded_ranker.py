"""Reproduce the frozen final CSV without rerunning the adaptive model search."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
from threadpoolctl import threadpool_limits

from final_ranker import feature_data, predict_ranker
from graded_ranker import fit_graded_ranker as fit_ranker
from sweettv.experiment import write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/reproduced_submission_v7.csv')
    args = parser.parse_args()
    run = Path('runs/v7_candidate')
    selection = json.loads((run / 'selection.json').read_text())
    cfg = json.loads(Path(selection['feature_config']).read_text())
    expected = json.loads((run / 'reproduction.json').read_text())['two_full_fits_identical_sha256']
    con = duckdb.connect('artifacts/data.duckdb', read_only=True)
    con.execute('SET threads=4')
    targets = [r[0] for r in con.execute('SELECT user_id FROM template').fetchall()]
    start, cutoff, end = '2020-03-09 00:00:00', '2020-07-27 00:00:00', '2020-10-19 00:00:00'
    with threadpool_limits(limits=4):
        ranker, _, audit = fit_ranker(con, start, cutoff, targets, cfg,
                                     selection['vocabulary'], selection['parameters'])
        data = feature_data(con, start, cutoff, end, targets, cfg, selection['vocabulary'])
        predictions = predict_ranker(data, targets, ranker)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        checks = write_submission('sweettv-tv-program-recommender/submission.csv',
                                  predictions, output, set(data['items'].tolist()))
    assert checks['sha256'] == expected, (checks['sha256'], expected)
    assert output.read_bytes() == (run / 'submission_v7.csv').read_bytes()
    print(json.dumps({'checks': checks, 'fit_audit': audit,
                      'matches_frozen_submission': True}, indent=2))


if __name__ == '__main__':
    main()
