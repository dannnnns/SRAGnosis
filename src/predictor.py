import base64
import io
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import CODE_LABELS, FEATURE_LABELS, FEATURES


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sragnosis_tree.joblib.b64"
METADATA_PATH = ROOT / "models" / "metadata.json"


def load_artifacts():
    model_bytes = base64.b64decode(MODEL_PATH.read_text(encoding="ascii"))
    model = joblib.load(io.BytesIO(model_bytes))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def case_frame(values):
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise ValueError(f"Campos ausentes: {', '.join(missing)}")
    return pd.DataFrame([{feature: str(values[feature]) for feature in FEATURES}])


def score_case(model, values):
    frame = case_frame(values)
    score = float(model.predict_proba(frame)[0, 1])
    return score


def score_band(score, screening_threshold):
    if score < screening_threshold:
        return "baixo", "Baixo sinal pelo modelo", "#16865C"
    if score < 0.5:
        return "intermediario", "Sinal intermediário", "#D97706"
    return "elevado", "Sinal elevado pelo modelo", "#C2414A"


def decision_path(model, values, maximum=6):
    frame = case_frame(values)
    transformed = model.named_steps["prep"].transform(frame)
    tree = model.named_steps["model"]
    encoded_names = model.named_steps["prep"].get_feature_names_out()
    nodes = tree.decision_path(transformed).indices
    leaf = tree.apply(transformed)[0]
    steps = []

    for node in nodes:
        if node == leaf:
            continue
        feature_index = tree.tree_.feature[node]
        if feature_index < 0:
            continue
        encoded_name = encoded_names[feature_index].replace("categorical__", "", 1)
        feature = next((name for name in FEATURES if encoded_name.startswith(f"{name}_")), None)
        if not feature:
            continue
        code = encoded_name[len(feature) + 1:]
        value = float(transformed[0, feature_index])
        category = CODE_LABELS.get(feature, {}).get(code, code)
        relation = "é" if value > tree.tree_.threshold[node] else "não é"
        statement = f"{FEATURE_LABELS[feature]} {relation} **{category}**"
        if statement not in steps:
            steps.append(statement)
        if len(steps) >= maximum:
            break
    return steps
