import io
import re
import unicodedata
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    TEM_XGB = True
except Exception:
    TEM_XGB = False

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Abandono Escolar e Trabalho Juvenil",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PASTA_DADOS = BASE_DIR / "dados"
SEED = 42

SIDRA_URL = (
    "https://apisidra.ibge.gov.br/values/t/4094/n2/all/v/4099,4723,12466,4090/p/"
    "201304,201404,201504,201604,201704,201804,201904,202004,202104,202204,"
    "202304,202404,202504/c58/114535,100052/h/y/f/a/d/m"
)

REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]
CANON = {
    "norte": "Norte", "nordeste": "Nordeste", "sudeste": "Sudeste",
    "sul": "Sul", "centro oeste": "Centro-Oeste", "brasil": "Brasil",
}
ETAPAS = {"Ensino Fundamental": "Abandono_EF", "Ensino Médio": "Abandono_EM"}
FEATURES_NUM = ["desocupacao_juvenil", "informalidade_juvenil"]
FEATURES_CAT = ["Regiao"]
FEATURES = FEATURES_NUM + FEATURES_CAT
TARGET = "Abandono_EM"


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def norm(x):
    """Minúsculas, sem acentos e sem espaços duplicados."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.lower().strip())


def canon(local):
    return CANON.get(norm(local).replace("-", " "), str(local).strip())


def nivel_de(local):
    if local == "Brasil":
        return "Brasil"
    return "Região" if local in REGIOES else "UF"


def para_numero(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# ----------------------------------------------------------------------------
# Leitura do INEP (arquivos .xlsx com cabeçalho em vários níveis)
# ----------------------------------------------------------------------------
def _parse_aba(raw, ano_arq):
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return None

    # 1) primeira linha de dados: um ano (ex.: 2023) nas primeiras colunas
    ini = None
    for i in range(min(40, len(raw))):
        for v in raw.iloc[i, :3].tolist():
            n = pd.to_numeric(v, errors="coerce")
            if pd.notna(n) and 2000 <= n <= 2030 and float(n).is_integer():
                ini = i
                break
        if ini is not None:
            break
    if ini is None:  # alternativa: primeira linha que contém "Brasil"
        for i in range(min(40, len(raw))):
            if any(norm(v) == "brasil" for v in raw.iloc[i, :6].tolist()):
                ini = i
                break
    if ini is None or ini == 0:
        return None

    # 2) nomes de colunas combinando os níveis do cabeçalho (ignora a linha de códigos NU_/NO_/1_CAT_)
    linhas = []
    for r in range(ini):
        vals = [norm(x) for x in raw.iloc[r].tolist()]
        nao_vazios = [v for v in vals if v]
        if not nao_vazios:
            continue
        if sum(1 for v in nao_vazios if "_" in v) >= max(2, len(nao_vazios) // 2):
            continue  # linha de códigos técnicos
        marcados = [k for k, v in enumerate(vals) if "taxa" in v or v.startswith("ensino")]
        if marcados and len(nao_vazios) >= 3:
            atual = ""
            for k in range(marcados[0], len(vals)):
                if vals[k]:
                    atual = vals[k]
                else:
                    vals[k] = atual
        linhas.append(vals)

    nomes = [" | ".join(l[c] for l in linhas if l[c]) for c in range(raw.shape[1])]

    def ultimo(n):
        return n.split(" | ")[-1]

    dados = raw.iloc[ini:].reset_index(drop=True)
    amostra = dados.head(300)
    ncols_texto = min(6, dados.shape[1])

    def valores(c):
        return set(amostra.iloc[:, c].map(norm))

    # 3) colunas de local / localização / dependência identificadas pelo CONTEÚDO
    geo_c, melhor = None, 0
    for c in range(ncols_texto):
        v = valores(c)
        if "brasil" in v and len(v) > melhor:
            geo_c, melhor = c, len(v)
    if geo_c is None:
        return None
    loc_c = next((c for c in range(ncols_texto) if "urbana" in valores(c)), None)
    dep_c = next((c for c in range(ncols_texto)
                  if {"federal", "municipal", "privada"} & valores(c)), None)

    # 4) colunas de abandono (total) por etapa
    def abandono(etapa):
        cand = [c for c, n in enumerate(nomes)
                if "abandono" in n and etapa in n and ultimo(n) == "total"]
        if not cand:
            cand = [c for c, n in enumerate(nomes)
                    if "abandono" in n and etapa in n and "total" in n]
        return cand[0] if cand else None

    c_ef, c_em = abandono("fundamental"), abandono("medio")
    if c_ef is None and c_em is None:
        return None

    def serie(c):
        return dados.iloc[:, c]

    mask = pd.Series(True, index=dados.index)
    if loc_c is not None:
        mask &= serie(loc_c).map(norm).eq("total")
    if dep_c is not None:
        mask &= serie(dep_c).map(norm).eq("total")

    out = pd.DataFrame({
        "Ano": ano_arq,
        "Local": serie(geo_c).astype(str).str.strip().map(canon),
        "Abandono_EF": para_numero(serie(c_ef)) if c_ef is not None else np.nan,
        "Abandono_EM": para_numero(serie(c_em)) if c_em is not None else np.nan,
    })[mask]
    out = out[~out["Local"].map(norm).isin(["", "nan"])]
    return out.dropna(subset=["Abandono_EF", "Abandono_EM"], how="all")

@st.cache_data(show_spinner="Lendo arquivos do INEP...")
def carregar_inep():
    arquivos = [f for f in sorted(PASTA_DADOS.glob("*.xlsx")) if not f.name.startswith("~$")]
    partes, diag = [], []
    for f in arquivos:
        m = re.search(r"(20\d{2})", f.name)
        ano = int(m.group(1)) if m else None
        resultado, aba_ok = None, None
        try:
            abas = pd.read_excel(f, sheet_name=None, header=None, engine="openpyxl")
            for nome, raw in abas.items():
                resultado = _parse_aba(raw, ano)
                if resultado is not None and len(resultado):
                    aba_ok = nome
                    break
        except Exception as e:
            diag.append({"Arquivo": f.name, "Ano": ano, "Status": f"erro: {e}", "Linhas": 0})
            continue
        if resultado is None or resultado.empty:
            diag.append({"Arquivo": f.name, "Ano": ano, "Status": "layout não reconhecido", "Linhas": 0})
        else:
            partes.append(resultado)
            diag.append({"Arquivo": f.name, "Ano": ano, "Status": f"ok (aba '{aba_ok}')",
                         "Linhas": len(resultado)})
    if not partes:
        return pd.DataFrame(), pd.DataFrame(diag)
    df = pd.concat(partes, ignore_index=True)
    df = df.dropna(subset=["Ano"]).drop_duplicates(subset=["Ano", "Local"])
    df["Ano"] = df["Ano"].astype(int)
    df["Nivel"] = df["Local"].map(nivel_de)
    return df.sort_values(["Ano", "Local"]).reset_index(drop=True), pd.DataFrame(diag)


# ----------------------------------------------------------------------------
# IBGE: CSV local -> API SIDRA -> upload
# ----------------------------------------------------------------------------
def achar_csv_ibge():
    for pasta in [PASTA_DADOS / "IBGE", PASTA_DADOS, BASE_DIR]:
        if pasta.exists():
            for f in sorted(pasta.glob("*IBGE*.csv")):
                return f
    return None


def preparar_ibge(bruto):
    df = bruto.copy()
    df["V"] = para_numero(df["V"])
    df["D2C"] = pd.to_numeric(df["D2C"], errors="coerce")
    df["Ano"] = pd.to_numeric(df["D3C"].astype(str).str[:4], errors="coerce")
    df = df[df["D2C"].isin([4099, 12466])].dropna(subset=["V", "Ano", "D1N", "D4N"])
    df["Indicador"] = df["D2C"].map({4099: "Taxa de desocupação", 12466: "Taxa de informalidade"})
    df["Regiao"] = df["D1N"].map(canon)
    df["Ano"] = df["Ano"].astype(int)
    return df[["Ano", "Regiao", "D4N", "Indicador", "V"]].reset_index(drop=True)


@st.cache_data(show_spinner="Carregando dados do IBGE...", ttl=3600)
def carregar_ibge(conteudo=None):
    try:
        if conteudo is not None:
            bruto = pd.read_csv(io.BytesIO(conteudo), encoding="utf-8-sig", low_memory=False)
            origem = "Arquivo enviado pelo usuário"
        else:
            f = achar_csv_ibge()
            if f is not None:
                bruto = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
                origem = f"Arquivo local ({f.name})"
            else:
                import requests
                r = requests.get(SIDRA_URL, timeout=25)
                r.raise_for_status()
                bruto = pd.DataFrame(r.json()[1:])
                origem = "API SIDRA/IBGE (consulta em tempo real)"
                try:
                    (PASTA_DADOS / "IBGE").mkdir(parents=True, exist_ok=True)
                    bruto.to_csv(PASTA_DADOS / "IBGE" / "IBGE_Trabalho_Juvenil_2013_2025.csv",
                                 index=False, encoding="utf-8-sig")
                except Exception:
                    pass
        ibge = preparar_ibge(bruto)
        if ibge.empty:
            return None, origem, "O arquivo não contém os indicadores 4099/12466."
        return ibge, origem, None
    except Exception as e:
        return None, None, str(e)


# ----------------------------------------------------------------------------
# Modelagem preditiva (Parte 2)
# ----------------------------------------------------------------------------
def make_preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([
        ("num", num, FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
    ])


def metricas(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R²": r2_score(y, p)}


@st.cache_resource(show_spinner="Treinando modelos (Random Forest, XGBoost e Rede Neural)...")
def treinar(df_ml, ano_corte):
    treino = df_ml[df_ml["Ano"] <= ano_corte]
    teste = df_ml[df_ml["Ano"] > ano_corte]
    X_tr, y_tr = treino[FEATURES], treino[TARGET]
    X_te, y_te = teste[FEATURES], teste[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    modelos, cv_rmse, params = {}, {}, {}

    base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    modelos["Baseline (média)"] = base

    rf = GridSearchCV(
        Pipeline([("prep", make_preprocessor()), ("model", RandomForestRegressor(random_state=SEED))]),
        {"model__n_estimators": [200, 500], "model__max_depth": [2, 3, None],
         "model__min_samples_leaf": [1, 2, 4]},
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1).fit(X_tr, y_tr)
    modelos["Random Forest"] = rf.best_estimator_
    cv_rmse["Random Forest"], params["Random Forest"] = -rf.best_score_, rf.best_params_

    if TEM_XGB:
        xgb = GridSearchCV(
            Pipeline([("prep", make_preprocessor()),
                      ("model", XGBRegressor(objective="reg:squarederror", random_state=SEED, n_jobs=1))]),
            {"model__n_estimators": [100, 300], "model__max_depth": [2, 3],
             "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8, 1.0]},
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1).fit(X_tr, y_tr)
        modelos["XGBoost"] = xgb.best_estimator_
        cv_rmse["XGBoost"], params["XGBoost"] = -xgb.best_score_, xgb.best_params_

    mlp = TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", make_preprocessor()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation="relu", alpha=1e-3,
                                   learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
                                   validation_fraction=0.2, n_iter_no_change=40, random_state=SEED)),
        ]),
        transformer=StandardScaler()).fit(X_tr, y_tr)
    modelos["Rede Neural (MLP)"] = mlp

    pred_te = {m: mod.predict(X_te) for m, mod in modelos.items()}
    pred_tr = {m: mod.predict(X_tr) for m, mod in modelos.items()}
    tab_te = pd.DataFrame({m: metricas(y_te, p) for m, p in pred_te.items()}).T
    tab_tr = pd.DataFrame({m: metricas(y_tr, p) for m, p in pred_tr.items()}).T

    imp = {}
    for m in ["Random Forest", "XGBoost"]:
        if m in modelos:
            r = permutation_importance(modelos[m], X_te, y_te, n_repeats=30, random_state=SEED,
                                       scoring="neg_root_mean_squared_error")
            imp[m] = pd.Series(r.importances_mean, index=FEATURES)

    return {"modelos": modelos, "pred_te": pred_te, "tab_te": tab_te, "tab_tr": tab_tr,
            "cv_rmse": cv_rmse, "params": params, "imp": pd.DataFrame(imp),
            "teste": teste.reset_index(drop=True), "n_tr": len(treino), "n_te": len(teste)}


# ----------------------------------------------------------------------------
# Carregamento dos dados
# ----------------------------------------------------------------------------
st.title("📚 Abandono Escolar e Trabalho Juvenil no Brasil")
st.caption("Checkpoint 3 — Dashboard integrando a análise descritiva (Parte 1) e a modelagem preditiva (Parte 2)")

inep, diag_inep = carregar_inep()
if inep.empty:
    st.error("Não foi possível ler os arquivos do INEP.")
    with st.expander("Diagnóstico da leitura", expanded=True):
        st.dataframe(diag_inep)
    arqs = sorted(PASTA_DADOS.glob("*2023*.xlsx")) or sorted(PASTA_DADOS.glob("*.xlsx"))
    if arqs:
        st.markdown(f"**Prévia bruta de `{arqs[0].name}`** (as 15 primeiras linhas de cada aba):")
        for nome, df_raw in pd.read_excel(arqs[0], sheet_name=None, header=None).items():
            st.write(f"Aba: {nome} — {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
            st.dataframe(df_raw.head(15).astype(str))
    st.stop()

st.sidebar.header("Dados do IBGE")
upload = st.sidebar.file_uploader("Enviar CSV do IBGE (opcional)", type="csv",
                                  help="Use caso o arquivo local não exista e a API esteja fora do ar.")
ibge, origem_ibge, erro_ibge = carregar_ibge(upload.getvalue() if upload else None)

# base integrada (Grande Região x Ano) para correlação e modelagem
df_ml = None
if ibge is not None:
    ibge_reg = ibge.groupby(["Ano", "Regiao", "Indicador"])["V"].mean().unstack("Indicador").reset_index()
    ibge_reg.columns.name = None
    ibge_reg = ibge_reg.rename(columns={"Taxa de desocupação": "desocupacao_juvenil",
                                        "Taxa de informalidade": "informalidade_juvenil"})
    for c in FEATURES_NUM:
        if c not in ibge_reg:
            ibge_reg[c] = np.nan
    reg = inep[inep["Nivel"] == "Região"].rename(columns={"Local": "Regiao"})
    df_ml = (reg.merge(ibge_reg[["Ano", "Regiao"] + FEATURES_NUM], on=["Ano", "Regiao"], how="inner")
             .dropna(subset=[TARGET]).reset_index(drop=True))

# ----------------------------------------------------------------------------
# Sidebar — filtros
# ----------------------------------------------------------------------------
st.sidebar.header("Filtros")
anos = sorted(inep["Ano"].unique().tolist())
if len(anos) > 1:
    a0, a1 = st.sidebar.slider("Período", anos[0], anos[-1], (anos[0], anos[-1]))
else:
    a0 = a1 = anos[0]
anos_per = [a for a in anos if a0 <= a <= a1]
ano_ref = st.sidebar.selectbox("Ano de referência (KPIs e rankings)", anos_per, index=len(anos_per) - 1)

locais = sorted(inep["Local"].unique().tolist(), key=lambda x: (nivel_de(x) != "Brasil", nivel_de(x) != "Região", x))
local_kpi = st.sidebar.selectbox("Local dos KPIs", locais, index=locais.index("Brasil") if "Brasil" in locais else 0)
padrao = [l for l in ["Brasil"] + REGIOES if l in locais]
locais_sel = st.sidebar.multiselect("Locais na comparação temporal", locais, default=padrao)
etapas_sel = st.sidebar.multiselect("Etapas de ensino", list(ETAPAS), default=list(ETAPAS))

with st.sidebar.expander("Integrantes do grupo"):
    st.markdown(
        "- Ana Luiza Santos Bertão — RM 563171\n- Anny Elly Pantoja Martin — RM 565055\n"
        "- Gisleine Muñoz Ticona — RM 563804\n- Larissa Machado Silva — RM 564168\n"
        "- Raira de Teves Costa — RM 564850\n- Sofia Bill Franken — RM 562767")

# ----------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------
st.subheader(f"Indicadores — {local_kpi}, {ano_ref}")


def kpi_inep(ano, local, col):
    d = inep[(inep["Ano"] == ano) & (inep["Local"] == local)][col]
    return float(d.iloc[0]) if len(d) and pd.notna(d.iloc[0]) else None


def ano_anterior(ano):
    ant = [a for a in anos if a < ano]
    return ant[-1] if ant else None


def kpi_ibge(ano, local, indicador):
    if ibge is None:
        return None
    d = ibge[(ibge["Ano"] == ano) & (ibge["Indicador"] == indicador)]
    if local != "Brasil":
        d = d[d["Regiao"] == local]
    v = d["V"].mean()
    return None if pd.isna(v) else float(v)


def metrica(coluna, rotulo, valor, anterior=None):
    if valor is None:
        coluna.metric(rotulo, "N/D")
    else:
        delta = None if anterior is None else f"{valor - anterior:+.2f} p.p. vs. ano anterior"
        coluna.metric(rotulo, f"{valor:.2f}%", delta, delta_color="inverse")


ant = ano_anterior(ano_ref)
c1, c2, c3, c4 = st.columns(4)
metrica(c1, "Abandono — Ensino Fundamental", kpi_inep(ano_ref, local_kpi, "Abandono_EF"),
        kpi_inep(ant, local_kpi, "Abandono_EF") if ant else None)
metrica(c2, "Abandono — Ensino Médio", kpi_inep(ano_ref, local_kpi, "Abandono_EM"),
        kpi_inep(ant, local_kpi, "Abandono_EM") if ant else None)
metrica(c3, "Desocupação juvenil", kpi_ibge(ano_ref, local_kpi, "Taxa de desocupação"))
metrica(c4, "Informalidade juvenil", kpi_ibge(ano_ref, local_kpi, "Taxa de informalidade"))
if ibge is None:
    st.warning("Indicadores do IBGE indisponíveis (veja a aba **Fontes e metodologia**).")
elif nivel_de(local_kpi) == "UF":
    st.caption("O IBGE (Tabela 4094) só traz dados por Grande Região, por isso N/D para UFs. "
               "Para o Brasil, é usada a média das cinco regiões.")

st.divider()

# ----------------------------------------------------------------------------
# Abas
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Abandono escolar", "💼 Mercado de trabalho", "🔗 Relação entre indicadores",
    "🤖 Modelagem preditiva", "📌 Fontes e metodologia"])

# ---- Aba 1 ------------------------------------------------------------------
with tab1:
    st.markdown("#### Evolução do abandono escolar")
    base = inep[inep["Ano"].between(a0, a1) & inep["Local"].isin(locais_sel)]
    if base.empty or not etapas_sel:
        st.warning("Selecione ao menos um local e uma etapa nos filtros.")
    else:
        longo = base.melt(id_vars=["Ano", "Local"], value_vars=[ETAPAS[e] for e in etapas_sel],
                          var_name="Etapa", value_name="Taxa (%)")
        longo["Etapa"] = longo["Etapa"].map({v: k for k, v in ETAPAS.items()})
        fig = px.line(longo, x="Ano", y="Taxa (%)", color="Local", facet_col="Etapa", markers=True)
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig)

    st.markdown("#### Ranking por localidade")
    r1, r2 = st.columns(2)
    nivel_rk = r1.radio("Recorte", ["Grandes Regiões", "UFs"], horizontal=True)
    etapa_rk = r2.radio("Etapa", list(ETAPAS), horizontal=True)
    col_rk = ETAPAS[etapa_rk]
    rk = inep[(inep["Ano"] == ano_ref) & (inep["Nivel"] == ("Região" if nivel_rk == "Grandes Regiões" else "UF"))]
    rk = rk.dropna(subset=[col_rk]).sort_values(col_rk)
    if rk.empty:
        st.info("Sem dados para esse recorte no ano selecionado.")
    else:
        fig = px.bar(rk, x=col_rk, y="Local", orientation="h", text_auto=".2f",
                     labels={col_rk: "Taxa de abandono (%)"},
                     title=f"Abandono — {etapa_rk} ({ano_ref})")
        fig.update_layout(height=max(350, 24 * len(rk)))
        st.plotly_chart(fig)

    st.markdown("#### Mapa de calor — Ensino Médio por ano e região")
    piv = (inep[inep["Nivel"].isin(["Brasil", "Região"]) & inep["Ano"].between(a0, a1)]
           .pivot_table(index="Local", columns="Ano", values="Abandono_EM"))
    if not piv.empty:
        st.plotly_chart(px.imshow(piv, text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
                                  labels={"color": "Abandono EM (%)"}))

    with st.expander("Ver e baixar os dados do INEP"):
        tabela = inep[inep["Ano"].between(a0, a1)]
        st.dataframe(tabela)
        st.download_button("Baixar CSV", tabela.to_csv(index=False).encode("utf-8-sig"),
                           "inep_abandono.csv", "text/csv")

# ---- Aba 2 ------------------------------------------------------------------
with tab2:
    if ibge is None:
        st.info("Dados do IBGE indisponíveis. Coloque o CSV em `dados/IBGE/`, envie-o pela barra lateral "
                "ou aguarde a API do SIDRA voltar.")
    else:
        st.markdown("#### Desocupação e informalidade juvenil por Grande Região")
        regs_ibge = sorted(ibge["Regiao"].unique().tolist())
        sel = st.multiselect("Regiões", regs_ibge, default=regs_ibge, key="reg_ibge")
        g = (ibge[ibge["Ano"].between(a0, a1) & ibge["Regiao"].isin(sel)]
             .groupby(["Ano", "Regiao", "Indicador"], as_index=False)["V"].mean())
        if g.empty:
            st.warning("Sem dados do IBGE para o período/regiões escolhidos.")
        else:
            fig = px.line(g, x="Ano", y="V", color="Regiao", facet_col="Indicador", markers=True,
                          labels={"V": "Taxa (%)"})
            fig.update_yaxes(matches=None)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(fig)

        st.markdown(f"#### Comparação entre faixas etárias ({ano_ref})")
        idade = (ibge[ibge["Ano"] == ano_ref].groupby(["D4N", "Indicador"], as_index=False)["V"].mean())
        if idade.empty:
            st.info(f"O IBGE não possui dados para {ano_ref} (a série não cobre 2020–2021).")
        else:
            st.plotly_chart(px.bar(idade, x="Indicador", y="V", color="D4N", barmode="group",
                                   text_auto=".1f", labels={"V": "Taxa (%)", "D4N": "Faixa etária"}))

# ---- Aba 3 ------------------------------------------------------------------
with tab3:
    if df_ml is None or df_ml.empty:
        st.info("A relação entre indicadores precisa dos dados do IBGE.")
    else:
        rel = df_ml[df_ml["Ano"].between(a0, a1)]
        st.markdown("#### Dispersão: abandono × mercado de trabalho (Grande Região × ano)")
        s1, s2 = st.columns(2)
        etapa_sc = s1.radio("Etapa", list(ETAPAS), horizontal=True, key="etapa_sc")
        ind_sc = s2.radio("Indicador do trabalho", ["Informalidade", "Desocupação"], horizontal=True)
        xcol = "informalidade_juvenil" if ind_sc == "Informalidade" else "desocupacao_juvenil"
        d = rel.dropna(subset=[xcol, ETAPAS[etapa_sc]])
        if len(d) < 3:
            st.warning("Poucos pontos para o período escolhido.")
        else:
            fig = px.scatter(d, x=xcol, y=ETAPAS[etapa_sc], color="Regiao", hover_data=["Ano"],
                             trendline="ols", trendline_scope="overall",
                             labels={xcol: f"{ind_sc} juvenil (%)", ETAPAS[etapa_sc]: f"Abandono {etapa_sc} (%)"})
            st.plotly_chart(fig)

        st.markdown("#### Matriz de correlação (gráfico estático)")
        cols = ["Abandono_EF", "Abandono_EM"] + FEATURES_NUM
        corr = rel[cols].corr()
        nomes = ["Abandono EF", "Abandono EM", "Desocupação", "Informalidade"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(nomes, rotation=30, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(nomes)
        for i in range(4):
            for j in range(4):
                v = corr.values[i, j]
                ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
        ax.set_title("Correlação de Pearson")
        st.pyplot(fig)
        st.caption("Correlação indica associação em nível agregado; não permite afirmar causalidade.")

# ---- Aba 4 ------------------------------------------------------------------
with tab4:
    st.markdown("#### Previsão do abandono no Ensino Médio por Grande Região")
    st.write("Alvo: `Abandono_EM`. Variáveis: desocupação juvenil, informalidade juvenil e região. "
             "Divisão **temporal**: treino até o ano de corte e teste nos anos seguintes. "
             "Todos os modelos são comparados a um *baseline* que sempre prevê a média do treino.")
    if df_ml is None or df_ml.empty:
        st.info("A modelagem precisa dos dados do IBGE.")
    else:
        anos_ml = sorted(df_ml["Ano"].unique().tolist())
        opcoes = [a for a in anos_ml if (df_ml["Ano"] <= a).sum() >= 15 and (df_ml["Ano"] > a).sum() >= 2]
        if not opcoes:
            st.warning("Poucas observações para dividir entre treino e teste.")
        else:
            corte = st.selectbox("Último ano de treino", opcoes, index=opcoes.index(2019) if 2019 in opcoes else 0)
            res = treinar(df_ml, corte)
            tab_te, tab_tr = res["tab_te"], res["tab_tr"]
            st.caption(f"Treino: {res['n_tr']} observações | Teste: {res['n_te']} observações "
                       f"(anos {sorted(res['teste']['Ano'].unique().tolist())})")
            if not TEM_XGB:
                st.warning("XGBoost não está instalado; o modelo foi omitido.")

            m1, m2 = st.columns(2)
            m1.markdown("**Desempenho no teste (anos não vistos)**")
            m1.dataframe(tab_te.round(3))
            m2.markdown("**Desempenho no treino**")
            m2.dataframe(tab_tr.round(3))

            comp = pd.concat([tab_te["RMSE"].rename("Teste"), tab_tr["RMSE"].rename("Treino")], axis=1)
            comp = comp.reset_index(names="Modelo").melt("Modelo", var_name="Conjunto", value_name="RMSE (p.p.)")
            st.plotly_chart(px.bar(comp, x="Modelo", y="RMSE (p.p.)", color="Conjunto", barmode="group",
                                   text_auto=".2f", title="RMSE por modelo (menor é melhor)"))

            ml_nomes = [m for m in tab_te.index if m != "Baseline (média)"]
            melhor = tab_te.loc[ml_nomes, "RMSE"].idxmin()
            rb, rm = tab_te.loc["Baseline (média)", "RMSE"], tab_te.loc[melhor, "RMSE"]
            ganho = (1 - rm / rb) * 100
            st.success(
                f"**Melhor modelo no teste: {melhor}** — RMSE {rm:.2f} p.p., MAE {tab_te.loc[melhor, 'MAE']:.2f} p.p., "
                f"R² {tab_te.loc[melhor, 'R²']:.2f}. Em relação ao baseline (RMSE {rb:.2f}), "
                f"o erro {'caiu' if ganho > 0 else 'aumentou'} {abs(ganho):.1f}%.")
            if ganho <= 0:
                st.warning("Nenhum modelo superou o baseline: neste recorte, os indicadores de trabalho "
                           "não se traduzem em previsão confiável.")

            st.markdown("#### Previsto × observado (teste)")
            linhas = []
            for m in ml_nomes:
                d = res["teste"][["Ano", "Regiao", TARGET]].copy()
                d["Modelo"], d["Previsto"] = m, res["pred_te"][m]
                linhas.append(d)
            dpv = pd.concat(linhas).rename(columns={TARGET: "Observado"})
            lo = float(min(dpv["Observado"].min(), dpv["Previsto"].min())) - 0.5
            hi = float(max(dpv["Observado"].max(), dpv["Previsto"].max())) + 0.5
            fig = px.scatter(dpv, x="Observado", y="Previsto", color="Regiao", facet_col="Modelo",
                             hover_data=["Ano"], range_x=[lo, hi], range_y=[lo, hi])
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="gray"),
                          row="all", col="all")
            st.plotly_chart(fig)

            if not res["imp"].empty:
                st.markdown("#### Importância por permutação (aumento do RMSE no teste)")
                imp = res["imp"].reset_index(names="Variável").melt("Variável", var_name="Modelo", value_name="Δ RMSE")
                st.plotly_chart(px.bar(imp, x="Δ RMSE", y="Variável", color="Modelo", barmode="group",
                                       orientation="h"))

            st.markdown("#### Simulador")
            sm1, sm2, sm3, sm4 = st.columns(4)
            modelo_sim = sm1.selectbox("Modelo", ml_nomes, index=ml_nomes.index(melhor))
            reg_sim = sm2.selectbox("Região", REGIOES)
            d_min, d_max = float(df_ml["desocupacao_juvenil"].min()), float(df_ml["desocupacao_juvenil"].max())
            des_sim = sm3.slider("Desocupação (%)", d_min, d_max, float(df_ml["desocupacao_juvenil"].median()))
            if df_ml["informalidade_juvenil"].notna().any():
                i_min, i_max = float(df_ml["informalidade_juvenil"].min()), float(df_ml["informalidade_juvenil"].max())
                inf_sim = sm4.slider("Informalidade (%)", i_min, i_max, float(df_ml["informalidade_juvenil"].median()))
            else:
                inf_sim = np.nan
            entrada = pd.DataFrame([{"desocupacao_juvenil": des_sim, "informalidade_juvenil": inf_sim,
                                     "Regiao": reg_sim}])[FEATURES]
            previsao = float(res["modelos"][modelo_sim].predict(entrada)[0])
            st.metric("Abandono previsto no Ensino Médio", f"{previsao:.2f}%")

            with st.expander("Hiperparâmetros escolhidos (validação cruzada)"):
                for m, p in res["params"].items():
                    st.write(f"**{m}** — RMSE CV: {res['cv_rmse'][m]:.3f}")
                    st.json(p)

# ---- Aba 5 ------------------------------------------------------------------
with tab5:
    st.markdown("""
#### Fonte dos dados
| Fonte | Descrição | Link |
|---|---|---|
| **INEP** | Taxas de Rendimento Escolar (2013–2023): abandono no Ensino Fundamental e Médio, por Brasil, Grande Região e UF | [gov.br/inep](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/taxas-de-rendimento-escolar/) |
| **IBGE/SIDRA** | Tabela 4094 (PNAD Contínua): taxa de desocupação (4099) e taxa de informalidade (12466) de jovens de 14–17 e 18–24 anos, por Grande Região | [apisidra.ibge.gov.br](https://apisidra.ibge.gov.br/values/t/4094) |

#### KPIs
- **Abandono EF / EM:** taxa de abandono total (%) por local e ano (recortes *Total* de localização e dependência administrativa).
- **Desocupação e informalidade juvenil:** média entre as faixas etárias de 14–17 e 18–24 anos (4º trimestre).

#### Metodologia
- Integração por **Ano + Grande Região**. A modelagem usa apenas as cinco regiões (poucas observações).
- Modelos: Random Forest e XGBoost (GridSearchCV, 5 folds) e rede neural MLP (3 camadas ocultas: 64-32-16, scikit-learn), comparados a um baseline.
- Divisão temporal treino/teste e pré-processamento (mediana, padronização, one-hot) dentro de `Pipeline`, sem vazamento.

#### Limitações
Dados agregados (não permitem causalidade nem conclusões sobre indivíduos), poucas observações,
IBGE sem 2020–2021 na série, e o período de teste (pós-pandemia) tem dinâmica diferente do treino.
""")
    st.markdown("#### Situação dos dados carregados")
    if ibge is not None:
        st.success(f"IBGE carregado — origem: {origem_ibge}")
    else:
        st.error(f"IBGE indisponível: {erro_ibge}")
    with st.expander("Diagnóstico da leitura dos arquivos do INEP"):
        st.dataframe(diag_inep)