"""Validate the archived runs and regenerate the seed-level results table."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics


def summarize(root: Path):
    manifest = json.loads((root / 'provenance.json').read_text())
    groups = defaultdict(list)
    for entry in manifest['runs']:
        path = root / 'runs' / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['published_sha256'], path.name
        run = json.loads(path.read_text())
        returns = run['eval_returns']
        assert len(returns) == 20, path.name
        assert all(math.isfinite(value) for value in returns), path.name
        assert math.isclose(statistics.mean(returns), run['greedy_score'], abs_tol=1e-7), path.name
        groups[run['env_name'], run['variant_name']].append(run)
    rows = []
    for (env, variant), runs in sorted(groups.items()):
        assert sorted(run['seed'] for run in runs) == [0, 1, 2], (env, variant)
        scores = [run['greedy_score'] for run in runs]
        success = [run['extra_metrics'].get('success_rate') for run in runs]
        rows.append({'environment': env, 'variant': variant, 'seeds': 3,
                     'mean_return': statistics.mean(scores), 'sample_sd': statistics.stdev(scores),
                     'success_rate': statistics.mean(success) if all(s is not None for s in success) else ''})
    return rows


def main():
    root = Path(__file__).resolve().parent / 'evidence'
    rows = summarize(root)
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Archived experiment results', '',
             '90 runs; three training seeds per configuration; 20 greedy evaluation episodes per run. '
             'The table reports the mean and sample standard deviation across the three seed means. '
             'These are descriptive statistics, not confidence intervals.', '',
             '| Environment | Variant | Mean return ± sample SD | Success rate |',
             '| --- | --- | ---: | ---: |']
    for row in rows:
        success = f"{row['success_rate']:.1%}" if row['success_rate'] != '' else '—'
        lines.append(f"| {row['environment']} | {row['variant']} | {row['mean_return']:.2f} ± {row['sample_sd']:.2f} | {success} |")
    lines += ['', '## Correction to the historical report', '',
              'The original 10-page course report is preserved as a historical submission. Its TreasureGrid narrative '
              'claims 100% success and near-optimal paths; the saved results contradict that claim. '
              'The actual configuration means range from 0% to 11.7% success. Pong remains near −21 return. '
              'Use this table and the raw runs for quantitative claims.', '',
              'Each run stores its configuration, budget, training trace and evaluation returns. '
              'Results have not been reproduced from distributed checkpoints; checkpoints are not included. '
              'See provenance.json for original and published SHA-256 hashes.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Validated {len(rows)} configurations / 90 runs; wrote evidence/summary.csv and RESULTS.md')


if __name__ == '__main__':
    main()
