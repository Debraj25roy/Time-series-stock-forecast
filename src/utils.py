from pathlib import Path
import joblib

def save_model(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)