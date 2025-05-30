import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = "runs"  # or use absolute path if needed

def get_all_runs():
    return sorted([
        d for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    ])

def inspect_tags():
    for run in get_all_runs():
        path = os.path.join(RUNS_DIR, run)
        try:
            event = EventAccumulator(path)
            event.Reload()
            tags = event.Tags().get('scalars', [])
            print(f"✅ {run}: {len(tags)} scalar tags")
            for tag in tags:
                print(f"   └─ {tag}")
        except Exception as e:
            print(f"❌ Error loading {run}: {e}")

if __name__ == "__main__":
    inspect_tags()
