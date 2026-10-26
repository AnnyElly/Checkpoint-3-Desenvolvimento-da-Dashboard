# Abandono Escolar e Trabalho Juvenil no Brasil — Dashboard (Checkpoint 3)

## 👥 Integrantes

| Nome | RM |

| Ana Luiza Santos Bertão | 563171 |
| Anny Elly Pantoja Martin | 565055 |
| Gisleine Muñoz Ticona | 563804 |
| Larissa Machado Silva | 564168 |
| Raira de Teves Costa | 564850 |
| Sofia Bill Franken | 562767 |

---

**Aplicação online:** https://checkpoint-3-desenvolvimento-da-dashboardgit-rqubkvmcsmeh8skrd.streamlit.app/

##  O que a dashboard apresenta

- **KPIs:** taxa de abandono no Ensino Fundamental e no Ensino Médio, taxa de desocupação juvenil e taxa de informalidade juvenil, com variação em relação ao ano anterior.
- **Gráficos interativos (Plotly):** evolução temporal, ranking por região/UF, mapa de calor, dispersão com linha de tendência e previsto × observado.
- **Gráfico estático (Matplotlib):** matriz de correlação entre os indicadores.
- **Filtros:** período, ano de referência, local, comparação entre locais e etapa de ensino.
- **Modelagem preditiva:** Random Forest, XGBoost e Rede Neural (MLP), comparados a um baseline, com métricas (MAE, RMSE, R²), importância das variáveis e simulador de previsão.
- **Fontes e metodologia:** documentação dos dados, KPIs e limitações.

---

## Fontes dos dados

| Fonte | Descrição |
|---|---|
| **INEP** | [Taxas de Rendimento Escolar (2013–2023)](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/taxas-de-rendimento-escolar/) |
| **IBGE / SIDRA** | [Tabela 4094 (PNAD Contínua)](https://apisidra.ibge.gov.br/values/t/4094): taxa de desocupação (código 4099) e taxa de informalidade (código 12466) de jovens de 14 a 17 e de 18 a 24 anos, por Grande Região |

---

## Instruções para execução local

### Pré-requisitos
- **Python 3.10 ou superior** ([download](https://www.python.org/downloads/))
- **Git** (opcional, para clonar o repositório)

### Passo a passo

**1. Clone o repositório** (ou baixe o ZIP pelo botão *Code → Download ZIP* e extraia):

```bash
git clone https://github.com/AnnyElly/Checkpoint-3-Desenvolvimento-da-Dashboard.git
cd CP06
```

**2. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**4. Execute a aplicação:**

```bash
streamlit run app.py
```

O navegador abrirá automaticamente em **http://localhost:8501**. 

