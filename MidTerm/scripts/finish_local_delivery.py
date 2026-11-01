"""After the last chat turn, copy the PDF and original session into a two-file folder.

This command never uploads anything. GitHub status comes from the separately
verified publication record; the final course upload is performed by the student.
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
    publication_path=Path('runs/final_review/publication.json')
    publication=json.loads(publication_path.read_text()) if publication_path.exists() else {}
    repo_url=publication.get('repo_url','https://github.com/dryjins/RecSys-LLMs')
    assert '0.44891' in text and repo_url.removeprefix('https://') in text
    published=publication.get('verified') is True
    if published:
        assert publication['submission_sha256']==outcome['sha256']
    assert reader.pages[4].images, 'Actual score screenshot must be embedded'
    if args.check_only:
        print(json.dumps({'pdf_pages': 6, 'best_private': outcome['private_score'],
                          'exact_csv_sha256': outcome['sha256'], 'screenshot_embedded': True,
                          'github_publication_verified': published, 'final_session_export_created': False}, indent=2))
        return
    folder = Path('output/delivery')
    report_name='nikolai_evdokimov_midterm_report.pdf'
    session_name='nikolai_evdokimov_midterm_session.jsonl'
    expected = {report_name,session_name}
    folder.mkdir(parents=True, exist_ok=True)
    assert {p.name for p in folder.iterdir()} <= expected, 'Unexpected files: preserve and review them'
    native = Path('output/session/original_ai_session.jsonl')
    subprocess.run([sys.executable, 'scripts/export_ai_session.py', '--output', str(native)], check=True)
    shutil.copy2(pdf, folder / report_name)
    shutil.copy2(native, folder / session_name)
    assert {p.name for p in folder.iterdir()} == expected
    assert (folder / session_name).read_bytes() == native.read_bytes()
    print('Two local files prepared in output/delivery. GitHub publication verified:',published)
    print('If this chat continues, refresh the original session export after its last turn.')
    if native.stat().st_size>10_000_000:
        print('HW FAQ requires contacting the instructor before submission for session logs over10MB. This command does not obtain that agreement.')


if __name__ == '__main__':
    main()
