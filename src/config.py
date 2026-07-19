FEATURES = [
    "PERD_OLFT",
    "PERD_PALA",
    "FEBRE",
    "SATURACAO",
    "FADIGA",
    "TOSSE",
    "GARGANTA",
    "CS_SEXO",
    "DIST_PRI_INTERN",
    "POSSUI_MORBIDADE",
    "FAIXA_ETARIA",
    "VACINAS",
]

FEATURE_LABELS = {
    "PERD_OLFT": "Perda de olfato",
    "PERD_PALA": "Perda de paladar",
    "FEBRE": "Febre",
    "SATURACAO": "Saturação de O₂ < 95%",
    "FADIGA": "Fadiga",
    "TOSSE": "Tosse",
    "GARGANTA": "Dor de garganta",
    "CS_SEXO": "Sexo",
    "DIST_PRI_INTERN": "Intervalo até a internação",
    "POSSUI_MORBIDADE": "Morbidades",
    "FAIXA_ETARIA": "Faixa etária",
    "VACINAS": "Histórico vacinal",
}

BINARY_OPTIONS = {
    "Sim": "1.0",
    "Não": "2.0",
    "Não informado": "9.0",
}

SEX_OPTIONS = {
    "Masculino": "1.0",
    "Feminino": "2.0",
}

# These labels reconstruct the categories used in the supplied processed dataset.
# They must be checked against the original preprocessing code before clinical use.
AGE_OPTIONS = {
    "0–4 anos": "1.0",
    "5–11 anos": "2.0",
    "12–17 anos": "3.0",
    "18–24 anos": "4.0",
    "25–29 anos": "5.0",
    "30–39 anos": "6.0",
    "40–49 anos": "7.0",
    "50–59 anos": "8.0",
    "60–69 anos": "9.0",
    "70–79 anos": "10.0",
    "80 anos ou mais": "11.0",
}

INTERVAL_OPTIONS = {
    "Até 7 dias": "1.0",
    "8–14 dias": "2.0",
    "15 dias ou mais": "3.0",
    "Não informado": "9.0",
}

MORBIDITY_OPTIONS = {
    "Nenhuma registrada": "0.0",
    "Uma morbidade": "2.0",
    "Múltiplas morbidades": "3.0",
    "Não informado": "9.0",
}

VACCINE_OPTIONS = {
    "Nenhuma registrada": "0.0",
    "Influenza apenas": "1.0",
    "COVID-19 apenas": "2.0",
    "Influenza e COVID-19": "3.0",
}

CODE_LABELS = {
    **{feature: {value: label for label, value in BINARY_OPTIONS.items()} for feature in [
        "PERD_OLFT", "PERD_PALA", "FEBRE", "SATURACAO", "FADIGA", "TOSSE", "GARGANTA"
    ]},
    "CS_SEXO": {value: label for label, value in SEX_OPTIONS.items()},
    "DIST_PRI_INTERN": {value: label for label, value in INTERVAL_OPTIONS.items()},
    "POSSUI_MORBIDADE": {value: label for label, value in MORBIDITY_OPTIONS.items()},
    "FAIXA_ETARIA": {value: label for label, value in AGE_OPTIONS.items()},
    "VACINAS": {value: label for label, value in VACCINE_OPTIONS.items()},
}
