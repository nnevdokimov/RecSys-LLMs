"""After the last chat turn, copy the PDF and original session into a two-file folder.

GitHub publication is a separate outstanding step. This command never uploads
anything and does not certify course readiness or remote repository contents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from pypdf import PdfReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    pdf = Path('output/pdf/nikolai_evdokimov_midterm_report_FINAL.pdf')
    run = Path('runs/v5_candidate')
    outcome = json.loads((run / 'kaggle_result.json').read_text())
    csv = run / 'submission_v5.csv'
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == outcome['sha256']
    assert outcome['private_score'] == 0.44891
    outcome_paths = [*Path('runs').glob('v*_candidate/kaggle_result.json'), Path('runs/v1_submitted/kaggle_result.json')]
    outcomes = [json.loads(p.read_text()) for p in outcome_paths]
    assert outcome['private_score'] == max(r['private_score'] for r in outcomes)
    reader = PdfReader(pdf)
    assert len(reader.pages) == 6
    text = '\n'.join(page.extract_text() for page in reader.pages)
    assert '0.44891' in text and 'github.com/dryjins/RecSys-LLMs' in text
    assert reader.pages[4].images, 'Actual score screenshot must be embedded'
    if args.check_only:
        print(json.dumps({'pdf_pages': 6, 'best_private': outcome['private_score'],
                          'exact_csv_sha256': outcome['sha256'], 'screenshot_embedded': True,
                          'github_publication_verified': False, 'final_session_export_created': False}, indent=2))
        return
    folder = Path('output/delivery')
    expected = {'report.pdf', 'original_ai_session.jsonl'}
    folder.mkdir(parents=True, exist_ok=True)
    assert {p.name for p in folder.iterdir()} <= expected, 'Unexpected files: preserve and review them'
    native = Path('output/session/original_ai_session.jsonl')
    subprocess.run([sys.executable, 'scripts/export_ai_session.py', '--output', str(native)], check=True)
    shutil.copy2(pdf, folder / 'report.pdf')
    shutil.copy2(native, folder / 'original_ai_session.jsonl')
    assert {p.name for p in folder.iterdir()} == expected
    assert (folder / 'original_ai_session.jsonl').read_bytes() == native.read_bytes()
    print('Two local files prepared in output/delivery. GitHub publication is still unverified.')
    print('If this chat continues, refresh the original session export after its last turn.')


if __name__ == '__main__':
    main()
