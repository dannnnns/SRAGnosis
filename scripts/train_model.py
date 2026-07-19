import argparse
import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.config import FEATURES  # noqa: E402


def evaluate(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "ppv": float(precision_score(y_true, prediction, zero_division=0)),
        "npv": float(tn / (tn + fn)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def build_pipeline():
    return Pipeline([
        ("prep", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURES),
        ])),
        ("model", DecisionTreeClassifier(
            criterion="entropy",
            max_depth=8,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=1,
        )),
    ])


def aggregate_importance(model):
    encoded_names = model.named_steps["prep"].get_feature_names_out()
    encoded_importance = model.named_steps["model"].feature_importances_
    importance = {feature: 0.0 for feature in FEATURES}
    for encoded_name, value in zip(encoded_names, encoded_importance):
        for feature in FEATURES:
            if encoded_name.startswith(f"categorical__{feature}_"):
                importance[feature] += float(value)
                break
    return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    data = data[data["ANO"].between(2020, 2023)].dropna(subset=["CLASSI_FIN"]).copy()
    data[FEATURES] = data[FEATURES].fillna(9).astype(str)
    data["CLASSI_FIN"] = data["CLASSI_FIN"].astype(int)

    train = data[data["ANO"].between(2020, 2021)]
    validation = data[data["ANO"] == 2022]
    test = data[data["ANO"] == 2023]

    model = build_pipeline()
    model.fit(train[FEATURES], train["CLASSI_FIN"])

    validation_probability = model.predict_proba(validation[FEATURES])[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        validation["CLASSI_FIN"], validation_probability
    )
    eligible = np.where(recall[:-1] >= 0.80)[0]
    threshold_index = eligible[np.argmax(precision[:-1][eligible])]
    screening_threshold = float(thresholds[threshold_index])

    test_probability = model.predict_proba(test[FEATURES])[:, 1]
    output_dir = ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_model_path = output_dir / "sragnosis_tree.joblib"
    joblib.dump(model, binary_model_path)
    (output_dir / "sragnosis_tree.joblib.b64").write_text(
        base64.b64encode(binary_model_path.read_bytes()).decode("ascii"), encoding="ascii"
    )

    metadata = {
        "model_name": "SRAGnosis admission-only decision tree",
        "version": "0.2.0-research",
        "outcome": "Classificação binária de COVID-19 entre registros de SRAG",
        "features": FEATURES,
        "training_period": "2020–2021",
        "validation_period": "2022",
        "test_period": "2023",
        "training_n": int(len(train)),
        "validation_n": int(len(validation)),
        "test_n": int(len(test)),
        "screening_threshold": screening_threshold,
        "hyperparameters": {
            "criterion": "entropy",
            "max_depth": 8,
            "min_samples_leaf": 100,
            "class_weight": "balanced",
            "random_state": 1,
        },
        "feature_importance": aggregate_importance(model),
        "validation_2022": evaluate(
            validation["CLASSI_FIN"], validation_probability, screening_threshold
        ),
        "test_2023": evaluate(test["CLASSI_FIN"], test_probability, screening_threshold),
        "limitations": [
            "Protótipo retrospectivo, sem validação clínica prospectiva ou externa.",
            "O escore da árvore não é uma probabilidade clínica calibrada.",
            "A prevalência e o perfil dos casos mudaram intensamente entre 2020 e 2023.",
            "Os rótulos das categorias reconstruídas precisam ser conferidos no pré-processamento original.",
            "Não utilizar para diagnóstico, exclusão de doença ou decisão terapêutica.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
