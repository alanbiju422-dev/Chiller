import os

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifacts')

def get_artifact_path(name: str) -> str:
    return os.path.join(ARTIFACTS_DIR, name)
