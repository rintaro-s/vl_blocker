"""
Simple training stub for RVC-like model.
This script exposes run_training(job_id, synthetic_id, data_dir, jobs_dict)
which simulates training and updates jobs_dict[job_id]['progress'] periodically.
"""
import time
from pathlib import Path


def run_training(job_id: str, synthetic_id: str, data_dir: str, jobs_dict: dict, epochs: int = 20):
    jobs_dict[job_id]['status'] = 'running'
    jobs_dict[job_id]['progress'] = 0

    data_dir = Path(data_dir)

    # simulate preprocessing time
    for i in range(3):
        time.sleep(0.5)
        jobs_dict[job_id]['progress'] = int((i + 1) * 2)

    # Simulate epoch training
    for e in range(epochs):
        # simulate per-epoch work
        time.sleep(0.2)
        jobs_dict[job_id]['progress'] = int(5 + (e + 1) * (90 / epochs))

    # finalize
    jobs_dict[job_id]['progress'] = 100
    jobs_dict[job_id]['status'] = 'done'
    jobs_dict[job_id]['result'] = {
        'model_path': str(Path('./models/rvc') / f"{synthetic_id}.pth")
    }
    # Create a small placeholder file so the system can detect a model exists.
    models_dir = Path('./models/rvc')
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"{synthetic_id}.pth"
    try:
        with open(model_file, 'wb') as f:
            f.write(b'PLACEHOLDER_PTH_FOR_' + synthetic_id.encode('utf-8'))
    except Exception:
        # Non-fatal: if we can't write, the jobs_dict still reports the intended path
        pass


if __name__ == '__main__':
    # Local run example
    import sys
    job_id = 'localtest'
    jobs = {job_id: {'status': 'queued'}}
    run_training(job_id, 'reimu', './data/uploads', jobs)
    print(jobs[job_id])
