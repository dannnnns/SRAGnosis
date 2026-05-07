# SRAGnosis: Predição de COVID-19 em Pacientes com SRAG

## 📌 Visão Geral
Este projeto apresenta um modelo de Machine Learning para a classificação de casos de COVID-19 utilizando a base de dados pública de Síndrome Respiratória Aguda Grave (SRAG) do OpenDATASUS (Brasil, 2013–2023). O objetivo é auxiliar na triagem diagnóstica baseada em perfis clínico-epidemiológicos.

## 🔬 Metodologia
Utilizamos o algoritmo **DecisionTreeClassifier** (Árvore de Decisão) para garantir a interpretabilidade dos resultados, permitindo que profissionais de saúde compreendam a lógica por trás da classificação.

### Configuração do Modelo (Hiperparâmetros):
- **Critério:** `entropy` (Ganho de Informação).
- **Profundidade Máxima (`max_depth`):** 4. *Decisão: Nível otimizado para manter a clareza visual da árvore e evitar overfitting.*
- **Divisão Mínima (`min_samples_split`):** 2.
- **Divisão de Dados:** 80% para treino e 20% para teste (com `random_state=1` para garantir reprodutibilidade).

## 📋 Variáveis Utilizadas (Features)
O modelo analisa as seguintes informações do paciente para realizar a predição:
- **Sintomas:** Perda de olfato (`PERD_OLFT`), Perda de paladar (`PERD_PALA`), Saturação de oxigênio (`SATURACAO`) e Tosse (`TOSSE`).
- **Histórico e Suporte:** Uso de suporte ventilatório (`SUPORT_VEN`), Internação em UTI (`UTI`) e Presença de morbidades (`POSSUI_MORBIDADE`).
- **Exames:** Resultados de Raio-X (`RAIOX_RES`) e Tomografia (`TOMO_RES`).
- **Demografia e Tempo:** Faixa etária, Ano do caso e Distância para internação (`DIST_PRI_INTERN`).
- **Prevenção:** Status vacinal (`VACINAS`).

## 📊 Performance do Modelo
- **Acurácia Global:** 91%
- **Visualizações:** O repositório inclui a árvore de decisão gerada (`srag_tree.png`), Matriz de Confusão e Curva ROC na pasta `/outputs`.

## 🚀 Como Reproduzir
1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
