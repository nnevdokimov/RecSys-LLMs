"""Copy the original native session byte-for-byte; run after the final work turn."""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_SOURCE = Path('/Users/nnevdokimov/.codex/sessions/2026/10/29/rollout-2026-10-29T21-33-22-01a188f0-266f-7ca3-8514-d0f566a02a85.jsonl')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=Path('output/session/original_ai_session.jsonl'))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Source must never be overwritten')
    data = args.source.read_bytes()
    # Validate only: never filter, reformat, truncate, redact or reconstruct records.
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    assert records and records[0]['type'] == 'session_meta'
    assert records[0]['payload']['id'] == '01a188f0-266f-7ca3-8514-d0f566a02a85'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    if args.source.read_bytes() != args.output.read_bytes():
        raise RuntimeError('Session changed during copy; rerun when work has stopped')
    manifest = {'source': str(args.source), 'output': str(args.output),
                'bytes': len(data), 'records': len(records), 'sha256': digest(data),
                'byte_identical_at_copy': True, 'transformed': False,
                'caveat': 'The source grows if this chat continues; rerun after the last turn. JSONL was not renamed to JSON.'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
