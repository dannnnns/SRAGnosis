from src.config import FEATURES
from src.predictor import case_frame, load_artifacts, score_band, score_case


def example_case():
    return {
        "PERD_OLFT": "2.0",
        "PERD_PALA": "2.0",
        "FEBRE": "1.0",
        "SATURACAO": "1.0",
        "FADIGA": "2.0",
        "TOSSE": "1.0",
        "GARGANTA": "2.0",
        "CS_SEXO": "1.0",
        "DIST_PRI_INTERN": "1.0",
        "POSSUI_MORBIDADE": "0.0",
        "FAIXA_ETARIA": "8.0",
        "VACINAS": "0.0",
    }


def test_case_has_expected_columns():
    assert list(case_frame(example_case()).columns) == FEATURES


def test_model_returns_bounded_score():
    model, metadata = load_artifacts()
    score = score_case(model, example_case())
    assert 0.0 <= score <= 1.0
    band, label, color = score_band(score, metadata["screening_threshold"])
    assert band in {"baixo", "intermediario", "elevado"}
    assert label
    assert color.startswith("#")
