import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from traffic_friction.collect import _remove_orphan_steps, STEP_COLUMNS


def test_resume_removes_only_uncommitted_steps():
    with TemporaryDirectory() as d:
        path = Path(d) / 'steps.csv'
        with path.open('w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
            w.writeheader()
            for episode in ('completed', 'incomplete'):
                for step in (1, 2):
                    row = {col: 0 for col in STEP_COLUMNS}
                    row.update(episode_id=episode, step=step)
                    w.writerow(row)
        assert _remove_orphan_steps(path, {'completed'}) == 2
        assert [r['episode_id'] for r in csv.DictReader(path.open())] == ['completed', 'completed']
