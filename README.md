# SRAGnosis

Protótipo acadêmico de interface para uma árvore de decisão que classifica retrospectivamente registros de Síndrome Respiratória Aguda Grave (SRAG) quanto à compatibilidade com COVID-19.

> **Atenção:** o SRAGnosis não é um dispositivo médico, não foi validado prospectivamente e não deve orientar diagnóstico, isolamento, tratamento ou qualquer decisão clínica.

## O que mudou nesta versão

O notebook inicial alcançava aproximadamente 91% de acurácia em uma divisão aleatória de registros de 2013–2023. A auditoria mostrou que o ano sozinho alcançava 83,6% de acurácia, pois todos os registros anteriores a 2020 eram negativos para COVID-19.

Esta versão:

- restringe a análise ao período pandêmico;
- remove ano, evolução, UTI, suporte ventilatório e exames de imagem;
- trata códigos como categorias com `OneHotEncoder`;
- treina em 2020–2021, seleciona o limiar em 2022 e testa em 2023;
- apresenta o resultado como escore experimental, não como diagnóstico.

## Executar a interface

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Reproduzir o treinamento

O CSV não é versionado no repositório. Para reconstruir o artefato:

```bash
python scripts/train_model.py /caminho/para/srag_dataset_2013_a_2023-attr_set_2.csv
```

O comando gera:

- `models/sragnosis_tree.joblib`
- `models/sragnosis_tree.joblib.b64`, usado pela aplicação para manter o artefato portátil
- `models/metadata.json`

## Desempenho temporal

No teste de 2023, o modelo obteve AUC ROC de aproximadamente 0,82. No limiar de triagem selecionado em 2022, a sensibilidade foi de aproximadamente 79%, a especificidade de 77% e o valor preditivo positivo de apenas 7%, refletindo a prevalência de 2,1% no conjunto de teste.

Esses resultados mostram discriminação moderada, mas também forte degradação de utilidade preditiva ao longo do tempo. A aplicação é adequada para demonstração metodológica e desenvolvimento acadêmico, não para uso assistencial.

## Limitação do codebook

Os rótulos de faixa etária, intervalo até internação, morbidades e vacinação foram reconstruídos a partir do banco já processado. Eles precisam ser conferidos com o código original de preparação dos dados antes de publicação formal.
