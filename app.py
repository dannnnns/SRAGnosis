import pandas as pd
import streamlit as st

from src.config import (
    AGE_OPTIONS,
    BINARY_OPTIONS,
    FEATURE_LABELS,
    INTERVAL_OPTIONS,
    MORBIDITY_OPTIONS,
    SEX_OPTIONS,
    VACCINE_OPTIONS,
)
from src.predictor import decision_path, load_artifacts, score_band, score_case


st.set_page_config(
    page_title="SRAGnosis | Protótipo de pesquisa",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#123c31,#16865c); color:white; margin-bottom: 1.1rem;}
    .hero h1 {margin: 0; font-size: 2.35rem; letter-spacing: -0.04em;}
    .hero p {margin: .45rem 0 0; color: #dcefe7; max-width: 760px;}
    .eyebrow {font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .85;}
    .result-card {padding: 1.4rem; border-radius: 18px; border: 1px solid #d9e8e1; background: white; box-shadow: 0 8px 30px rgba(25,73,58,.08);}
    .result-label {font-size: 1.35rem; font-weight: 750; margin-bottom: .25rem;}
    .score {font-size: 3rem; line-height: 1; font-weight: 800; letter-spacing: -.05em;}
    .small-note {font-size: .86rem; color: #51665f;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #dfeae5; padding: .8rem 1rem; border-radius: 14px;}
    div[data-testid="stForm"] {background: white; border: 1px solid #dfeae5; padding: 1.15rem; border-radius: 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def artifacts():
    return load_artifacts()


model, metadata = artifacts()
threshold = metadata["screening_threshold"]

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">SRAGnosis • versão de pesquisa</div>
      <h1>Triagem clínico-epidemiológica</h1>
      <p>Explore como uma árvore de decisão classifica retrospectivamente perfis de SRAG. O resultado é experimental e não substitui teste diagnóstico ou avaliação profissional.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

analysis_tab, performance_tab, method_tab = st.tabs([
    "Analisar um perfil", "Desempenho", "Método e limites"
])

with analysis_tab:
    st.warning(
        "Ferramenta exclusivamente acadêmica. Não use o resultado para excluir COVID-19, "
        "definir isolamento, prescrever tratamento ou tomar qualquer decisão clínica."
    )

    with st.form("case_form"):
        st.subheader("Dados disponíveis na admissão")
        profile_col, symptoms_col = st.columns([1, 1.35], gap="large")

        with profile_col:
            age = st.selectbox("Faixa etária", AGE_OPTIONS)
            sex = st.selectbox("Sexo", SEX_OPTIONS)
            interval = st.selectbox("Tempo entre início dos sintomas e internação", INTERVAL_OPTIONS)
            morbidity = st.selectbox("Morbidades registradas", MORBIDITY_OPTIONS)
            vaccines = st.selectbox("Histórico vacinal", VACCINE_OPTIONS)

        with symptoms_col:
            st.markdown("**Sinais e sintomas**")
            symptom_left, symptom_right = st.columns(2)
            with symptom_left:
                fever = st.selectbox("Febre", BINARY_OPTIONS, index=1)
                cough = st.selectbox("Tosse", BINARY_OPTIONS, index=1)
                saturation = st.selectbox("Saturação de O₂ < 95%", BINARY_OPTIONS, index=1)
                fatigue = st.selectbox("Fadiga", BINARY_OPTIONS, index=1)
            with symptom_right:
                smell = st.selectbox("Perda de olfato", BINARY_OPTIONS, index=1)
                taste = st.selectbox("Perda de paladar", BINARY_OPTIONS, index=1)
                throat = st.selectbox("Dor de garganta", BINARY_OPTIONS, index=1)

        submitted = st.form_submit_button("Analisar perfil", type="primary", width="stretch")

    if submitted:
        values = {
            "PERD_OLFT": BINARY_OPTIONS[smell],
            "PERD_PALA": BINARY_OPTIONS[taste],
            "FEBRE": BINARY_OPTIONS[fever],
            "SATURACAO": BINARY_OPTIONS[saturation],
            "FADIGA": BINARY_OPTIONS[fatigue],
            "TOSSE": BINARY_OPTIONS[cough],
            "GARGANTA": BINARY_OPTIONS[throat],
            "CS_SEXO": SEX_OPTIONS[sex],
            "DIST_PRI_INTERN": INTERVAL_OPTIONS[interval],
            "POSSUI_MORBIDADE": MORBIDITY_OPTIONS[morbidity],
            "FAIXA_ETARIA": AGE_OPTIONS[age],
            "VACINAS": VACCINE_OPTIONS[vaccines],
        }
        score = score_case(model, values)
        band, label, color = score_band(score, threshold)
        steps = decision_path(model, values)

        st.divider()
        result_col, context_col = st.columns([1, 1.15], gap="large")
        with result_col:
            st.markdown(
                f"""
                <div class="result-card" style="border-top: 5px solid {color}">
                  <div class="eyebrow" style="color:{color}">Resultado experimental</div>
                  <div class="result-label">{label}</div>
                  <div class="score" style="color:{color}">{score:.0%}</div>
                  <div class="small-note">escore produzido pela árvore; não é probabilidade clínica calibrada</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(int(score * 100), 100))
            if band == "baixo":
                st.info("O escore ficou abaixo do limiar experimental de triagem. Isso não exclui COVID-19.")
            else:
                st.warning("O escore atingiu o limiar experimental. O resultado não confirma COVID-19 e exige investigação apropriada.")

        with context_col:
            st.markdown("#### Como a árvore chegou aqui")
            if steps:
                for step in steps:
                    st.markdown(f"- {step}")
            st.caption(
                "A sequência mostra condições do caminho decisório, não relações causais nem fatores de risco independentes."
            )

        with st.expander("Por que este alerta pode errar?"):
            st.markdown(
                "Em 2023, apenas 2,1% dos registros do conjunto de teste eram positivos. "
                "No limiar usado nesta tela, a sensibilidade foi 79,1%, mas o valor preditivo "
                "positivo foi 7,0%: a maioria dos alertas positivos foi falso-positiva."
            )

with performance_tab:
    st.subheader("Validação em anos posteriores")
    st.write(
        "O modelo foi ajustado em 2020–2021. O limiar foi escolhido em 2022 para priorizar "
        "sensibilidade e depois aplicado, sem reajuste, aos registros de 2023."
    )
    validation = metadata["validation_2022"]
    test = metadata["test_2023"]

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("AUC ROC • 2023", f"{test['roc_auc']:.3f}")
    col_2.metric("Sensibilidade • 2023", f"{test['sensitivity']:.1%}")
    col_3.metric("Especificidade • 2023", f"{test['specificity']:.1%}")
    col_4.metric("VPP • 2023", f"{test['ppv']:.1%}")

    comparison = pd.DataFrame({
        "Métrica": ["Prevalência", "Sensibilidade", "Especificidade", "VPP", "VPN", "AUC ROC", "AUC PR"],
        "Validação 2022": [
            validation["prevalence"], validation["sensitivity"], validation["specificity"],
            validation["ppv"], validation["npv"], validation["roc_auc"], validation["pr_auc"],
        ],
        "Teste temporal 2023": [
            test["prevalence"], test["sensitivity"], test["specificity"],
            test["ppv"], test["npv"], test["roc_auc"], test["pr_auc"],
        ],
    }).set_index("Métrica")
    st.dataframe(comparison.style.format("{:.1%}"), width="stretch")

    st.markdown("#### Importância global das variáveis")
    importance = pd.Series(metadata["feature_importance"]).rename(index=FEATURE_LABELS)
    st.bar_chart(importance.sort_values(ascending=True), horizontal=True)
    st.caption("Importância da árvore no conjunto de treinamento; não representa causalidade.")

with method_tab:
    st.subheader("O que foi mudado em relação ao notebook original")
    st.markdown(
        """
        - Foram mantidos apenas registros de 2020–2023 para evitar que anos pré-pandemia funcionassem como resposta pronta.
        - `ANO`, `EVOLUCAO`, `UTI`, `SUPORT_VEN`, `RAIOX_RES` e `TOMO_RES` foram retirados do formulário e do modelo.
        - Os códigos foram tratados como categorias, e não como números contínuos ou ordinais.
        - O treinamento, a seleção do limiar e o teste foram separados cronologicamente.
        - A árvore usa profundidade máxima 8, mínimo de 100 observações por folha e ponderação de classes.
        """
    )

    st.markdown("#### Limites que ainda permanecem")
    for limitation in metadata["limitations"]:
        st.markdown(f"- {limitation}")

    st.error(
        "As categorias de faixa etária, intervalo, morbidades e vacinação foram reconstruídas "
        "a partir do banco processado. Antes de qualquer publicação formal, elas precisam ser "
        "conferidas com o código original de preparação dos dados."
    )

st.caption(
    f"SRAGnosis {metadata['version']} • treinamento: {metadata['training_period']} • "
    f"validação: {metadata['validation_period']} • teste temporal: {metadata['test_period']}"
)
