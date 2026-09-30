"""
Coletor CVM — Peers Long Bias x AWR Long Bias
=============================================
Baixa (com cache) e processa, desde a abertura do AWR (10/04/2025):
  - Informe Diário (cota, PL, captação, resgate, cotistas)
  - CDA mensal (carteira), com look-through FIC -> master -> ...
  - Preços mensais das ações (yfinance) + Ibovespa + CDI (BCB)
e grava arquivos compactos em data/ que o dashboard (app.py) lê.

Rodar:  python coletor_cvm.py            (usa cache; rebaixa CDA dos últimos 8 meses)
        python coletor_cvm.py --refresh  (rebaixa tudo)
"""
import io
import re
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
DATA = BASE / "data"
RAW.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

INICIO = date(2025, 4, 10)          # abertura do AWR Long Bias
AWR = "60.171.849/0001-50"

PEERS = {
    "28.747.685/0001-53": "Kapitalo Tarkus",
    "38.971.881/0001-60": "Truxt Long Bias",
    "13.277.011/0001-65": "Opportunity Log",
    "46.479.577/0001-29": "Itaú Optimus LB",
    "15.334.585/0001-53": "SPX Patriot",
    "73.232.530/0001-39": "Dynamo Cougar",
    AWR:                  "AWR Long Bias",
    "46.098.790/0001-90": "Absolute Pace",
    "59.965.040/0001-10": "Navi Long Biased",
    "12.823.624/0001-98": "Oceana Long Biased",
    "34.839.385/0001-05": "Alphakey Ações",
    "35.744.266/0001-23": "Constellation F",
    "10.500.884/0001-05": "Real Investor",
    "09.285.146/0001-03": "Squadra Long Biased",
    "37.487.351/0001-89": "Encore Long Bias",
}

URL_CDA = "https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_{ym}.zip"
URL_INF = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ym}.zip"
URL_CDI = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={ini}&dataFinal={fim}"

# ETFs não são abertos (viram um ativo único)
RE_ETF = re.compile(r"ÍNDICE|INDICE|\bETF\b|ISHARES", re.I)
RE_RF = re.compile(r"RENDA FIXA|REFERENCIADO|\bRF\b|\bDI\b|SOBERANO|TPF|TESOURO|CAIXA FIF|ZERAGEM|CURTO PRAZO", re.I)
ETF_TICKER = {
    "ISHARES BM&FBOVESPA SMALL CAP": "SMAL11",
    "ISHARES IBOVESPA": "BOVA11",
    "IT NOW IBOVESPA": "BOVX11",
}


def meses(ini: date, fim: date):
    y, m = ini.year, ini.month
    while (y, m) <= (fim.year, fim.month):
        yield f"{y}{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def baixar(url: str, destino: Path, forcar: bool = False) -> Path | None:
    if destino.exists() and not forcar:
        return destino
    r = requests.get(url, timeout=300)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    destino.write_bytes(r.content)
    print(f"  baixado {destino.name} ({len(r.content)/1e6:.1f} MB)")
    return destino


def ler(z: zipfile.ZipFile, prefixo: str, usecols=None) -> pd.DataFrame:
    nome = next((n for n in z.namelist() if n.startswith(prefixo)), None)
    if nome is None:
        return pd.DataFrame(columns=usecols or [])
    df = pd.read_csv(z.open(nome), sep=";", encoding="latin1", dtype=str,
                     usecols=lambda c: usecols is None or c in usecols, low_memory=False)
    for c in df.columns:
        if c.startswith(("VL_", "QT_")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


TIPOS_ACAO_LONG = ["Ação", "Ação (emprestada)", "BDR", "Recibo", "Exterior", "ETF", "Termo"]


def classificar_b4(tp_aplic: str) -> tuple[str, int]:
    """TP_APLIC do bloco 4 -> (tipo, sinal da exposição)."""
    t = tp_aplic.lower()
    if t.startswith("obrigações por ações"):
        return "Vendido (short)", -1
    if "cedidos em empr" in t:
        return "Ação (emprestada)", 1
    if t == "ações":
        return "Ação", 1
    if "bdr" in t:
        return "BDR", 1
    if t.startswith("certificado ou recibo"):
        return "Recibo", 1
    if "opções - posições titulares" in t:
        return "Opção comprada", 1
    if "opções - posições lançadas" in t:
        return "Opção lançada", -1
    if "compras a termo" in t:
        return "Termo", 1
    if "vendas a termo" in t:
        return "Termo vendido", -1
    if "futuro" in t:
        return "Futuro", 1
    if "debênture" in t:
        return "Crédito", 1
    return "Outros", 1


def etf_ticker(nome: str) -> str:
    nome_up = (nome or "").upper()
    for k, v in ETF_TICKER.items():
        if k in nome_up:
            return v
    return "ETF " + nome_up[:30].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1) Informe diário
# ─────────────────────────────────────────────────────────────────────────────
def coletar_inf_diario(forcar: bool):
    frames = []
    hoje = date.today()
    for ym in meses(INICIO, hoje):
        p = RAW / f"inf_diario_fi_{ym}.zip"
        velho = p.exists() and (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() > 6 * 3600
        recente = ym >= (hoje - timedelta(days=62)).strftime("%Y%m")
        p = baixar(URL_INF.format(ym=ym), p, forcar or (recente and velho))
        if p is None:
            continue
        with zipfile.ZipFile(p) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), sep=";", encoding="latin1", dtype=str)
        df = df[df["CNPJ_FUNDO_CLASSE"].isin(PEERS)]
        frames.append(df)
        print(f"  inf_diario {ym}: {len(df)} linhas")
    df = pd.concat(frames, ignore_index=True)
    for c in ["VL_TOTAL", "VL_QUOTA", "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"])
    df["ID_SUBCLASSE"] = df["ID_SUBCLASSE"].fillna("")
    df = df.drop_duplicates(["CNPJ_FUNDO_CLASSE", "ID_SUBCLASSE", "DT_COMPTC"], keep="last")

    # cota = série emendada pela subclasse de maior PL em cada data (fundos migram de
    # "sem subclasse" para subclasses no meio da série); PL/fluxos = soma das subclasses
    out = []
    for cnpj, g in df.groupby("CNPJ_FUNDO_CLASSE"):
        cotas_sub = g.pivot_table(index="DT_COMPTC", columns="ID_SUBCLASSE", values="VL_QUOTA").sort_index()
        pl_sub = g.pivot_table(index="DT_COMPTC", columns="ID_SUBCLASSE", values="VL_PATRIM_LIQ").reindex(cotas_sub.index)
        escolhida = pl_sub.fillna(-1).idxmax(axis=1)
        rets, ant = [], None
        for i, (dt, s) in enumerate(escolhida.items()):
            if i == 0:
                rets.append(0.0)
            else:
                prev = cotas_sub.index[i - 1]
                if pd.notna(cotas_sub.at[prev, s]):
                    r = cotas_sub.at[dt, s] / cotas_sub.at[prev, s] - 1
                elif ant is not None and pd.notna(cotas_sub.at[prev, ant]):
                    # troca de série: aceita emenda de nível só se for a mesma linhagem de cota
                    r = cotas_sub.at[dt, s] / cotas_sub.at[prev, ant] - 1
                    r = r if abs(r) < 0.05 else 0.0
                else:
                    r = 0.0
                rets.append(r)
            ant = s
        cota = pd.Series(np.cumprod(1 + np.array(rets)), index=cotas_sub.index)
        agg = g.groupby("DT_COMPTC")[["VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST"]].sum()
        agg["cota"] = cota
        agg["cnpj"] = cnpj
        out.append(agg.reset_index())
    diario = pd.concat(out, ignore_index=True).rename(columns={
        "DT_COMPTC": "data", "VL_PATRIM_LIQ": "pl", "CAPTC_DIA": "captacao",
        "RESG_DIA": "resgate", "NR_COTST": "cotistas"})
    diario = diario[diario["data"] >= pd.Timestamp(INICIO)].sort_values(["cnpj", "data"])
    diario["fundo"] = diario["cnpj"].map(PEERS)
    diario.to_parquet(DATA / "diario.parquet", index=False)
    print(f"-> diario.parquet: {len(diario)} linhas, até {diario['data'].max().date()}")
    return diario


# ─────────────────────────────────────────────────────────────────────────────
# 2) CDA mensal com look-through
# ─────────────────────────────────────────────────────────────────────────────
COLS_BASE = ["CNPJ_FUNDO_CLASSE", "DT_COMPTC", "TP_APLIC", "TP_ATIVO", "QT_POS_FINAL",
             "VL_MERC_POS_FINAL", "VL_VENDA_NEGOC", "VL_AQUIS_NEGOC"]


def processar_cda_mes(path: Path):
    """Retorna dicionários brutos do mês (apenas fundos relevantes)."""
    with zipfile.ZipFile(path) as z:
        pl = ler(z, "cda_fi_PL", ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "VL_PATRIM_LIQ"])
        b2 = ler(z, "cda_fi_BLC_2", COLS_BASE + ["CNPJ_FUNDO_CLASSE_COTA", "NM_FUNDO_CLASSE_SUBCLASSE_COTA"])
        conf = ler(z, "cda_fi_CONFID", ["CNPJ_FUNDO_CLASSE", "TP_APLIC", "VL_MERC_POS_FINAL"])

        # fecho transitivo: peers + tudo que eles (e seus masters) carregam em cotas
        alvo = set(PEERS)
        while True:
            novos = set(b2.loc[b2["CNPJ_FUNDO_CLASSE"].isin(alvo), "CNPJ_FUNDO_CLASSE_COTA"].dropna()) - alvo
            if not novos:
                break
            alvo |= novos
        # masters conhecidos de meses anteriores também entram (vínculo confidencial)
        alvo |= set(LINKS_CONHECIDOS.get("_todos", set()))

        b4 = ler(z, "cda_fi_BLC_4", COLS_BASE + ["CD_ATIVO", "DS_ATIVO"])
        # preço de fechamento implícito (VL/QT) usando todos os fundos do mercado
        px = b4[b4["TP_APLIC"].isin(["Ações", "Ações e outros TVM cedidos em empréstimo",
                                     "Brazilian Depository Receipt - BDR",
                                     "Certificado ou recibo de depósito de valores mobiliários"])
                & (b4["QT_POS_FINAL"] > 0) & (b4["VL_MERC_POS_FINAL"] > 0)].copy()
        px["preco"] = px["VL_MERC_POS_FINAL"] / px["QT_POS_FINAL"]
        precos = px.groupby(px["CD_ATIVO"].str.strip().str.upper())["preco"].agg(["median", "count"])
        b4 = b4[b4["CNPJ_FUNDO_CLASSE"].isin(alvo)]
        b7 = ler(z, "cda_fi_BLC_7", COLS_BASE + ["CD_ATIVO_BV_MERC", "DS_ATIVO_EXTERIOR", "EMISSOR"])
        b7 = b7[b7["CNPJ_FUNDO_CLASSE"].isin(alvo)]

    pl = pl[pl["CNPJ_FUNDO_CLASSE"].isin(alvo)].drop_duplicates("CNPJ_FUNDO_CLASSE")
    b2 = b2[b2["CNPJ_FUNDO_CLASSE"].isin(alvo)]
    conf = conf[conf["CNPJ_FUNDO_CLASSE"].isin(alvo)]
    return pl, b2, b4, b7, conf, precos


LINKS_CONHECIDOS: dict = {}   # holder -> {held: fração do valor em cotas}


def montar_exposicoes(ym, pl, b2, b4, b7, conf):
    """Explode a carteira de cada peer até o nível de ativo (look-through)."""
    plmap = pl.set_index("CNPJ_FUNDO_CLASSE")["VL_PATRIM_LIQ"].to_dict()
    nomes = pl.set_index("CNPJ_FUNDO_CLASSE")["DENOM_SOCIAL"].to_dict()

    # posições diretas por fundo: lista (ativo, tipo, valor, desc, aquis, venda)
    diretas: dict[str, list] = {}
    for _, r in b4.iterrows():
        tk = str(r["CD_ATIVO"] if pd.notna(r["CD_ATIVO"]) else "").strip().upper() or str(r["DS_ATIVO"])[:20]
        tipo, sinal = classificar_b4(str(r["TP_APLIC"]))
        v = abs(r["VL_MERC_POS_FINAL"] or 0.0) * sinal if tipo != "Futuro" else (r["VL_MERC_POS_FINAL"] or 0.0)
        diretas.setdefault(r["CNPJ_FUNDO_CLASSE"], []).append(
            (tk, tipo, v, str(r["DS_ATIVO"] if pd.notna(r["DS_ATIVO"]) else "").strip(),
             r["VL_AQUIS_NEGOC"] or 0.0, r["VL_VENDA_NEGOC"] or 0.0))
    for _, r in b7.iterrows():
        ta = str(r["TP_ATIVO"])
        emissor = str(r["EMISSOR"] if pd.notna(r["EMISSOR"]) else "").strip()
        ds = str(r["DS_ATIVO_EXTERIOR"] if pd.notna(r["DS_ATIVO_EXTERIOR"]) else "").strip()
        if ta == "Fundos Offshore":
            tk, tipo = "OFFSHORE " + emissor.split(" - ")[1][:25] if " - " in emissor else "OFFSHORE", "Offshore"
        elif ta.startswith("Ação") or "Depository" in ta:
            tk, tipo = (ds or emissor or "EXTERIOR")[:25].upper(), "Exterior"
        else:
            tk, tipo = "Exterior - " + ta[:25], "Exterior outros"
        diretas.setdefault(r["CNPJ_FUNDO_CLASSE"], []).append(
            (tk, tipo, r["VL_MERC_POS_FINAL"] or 0.0, f"{ta}: {ds or emissor}"[:80],
             r["VL_AQUIS_NEGOC"] or 0.0, r["VL_VENDA_NEGOC"] or 0.0))

    # cotas de fundos (abertas) e confidenciais
    cotas: dict[str, list] = {}
    for _, r in b2.dropna(subset=["CNPJ_FUNDO_CLASSE_COTA"]).iterrows():
        cotas.setdefault(r["CNPJ_FUNDO_CLASSE"], []).append(
            (r["CNPJ_FUNDO_CLASSE_COTA"], r["VL_MERC_POS_FINAL"], str(r["NM_FUNDO_CLASSE_SUBCLASSE_COTA"] or "")))
    confid = conf.groupby(["CNPJ_FUNDO_CLASSE", "TP_APLIC"])["VL_MERC_POS_FINAL"].sum()

    # aprende vínculos abertos (para usar em meses em que ficam confidenciais)
    for h, lst in cotas.items():
        tot = sum(v for _, v, _ in lst if v and v > 0)
        if tot > 0:
            LINKS_CONHECIDOS[h] = {c: v / tot for c, v, _ in lst if v and v > 0}
            LINKS_CONHECIDOS.setdefault("_nomes", {}).update({c: n for c, _, n in lst})
            LINKS_CONHECIDOS.setdefault("_todos", set()).update(c for c, _, _ in lst)

    def explodir(fundo, fator, prof=0):
        """yield (ativo, tipo, valor_proporcional, desc, aquis, venda)"""
        if prof > 5:
            return
        for tk, tipo, v, d, a, s in diretas.get(fundo, []):
            yield tk, tipo, v * fator, d, a * fator, s * fator
        abertos = cotas.get(fundo, [])
        for held, v, nome in abertos:
            if RE_ETF.search(nome) or RE_ETF.search(nomes.get(held, "")):
                yield etf_ticker(nome or nomes.get(held, "")), "ETF", v * fator, nome, 0, 0
            elif RE_RF.search(nome or nomes.get(held, "")):
                yield "Fundo RF/Caixa", "Caixa/RF", v * fator, nome, 0, 0
            elif held in plmap and plmap[held] and (held in diretas or held in cotas or
                                                    held in confid.index.get_level_values(0)):
                yield from explodir(held, fator * v / plmap[held], prof + 1)
            else:
                yield "Outros fundos", "Fundo", v * fator, nome, 0, 0
        # confidenciais
        for (f, tp), v in confid.items():
            if f != fundo or not v:
                continue
            if tp == "Cotas de Fundos" and not abertos and fundo in LINKS_CONHECIDOS:
                for held, frac in LINKS_CONHECIDOS[fundo].items():
                    if held in plmap and plmap[held]:
                        yield from explodir(held, fator * v * frac / plmap[held], prof + 1)
                    else:
                        yield "CONFIDENCIAL", "Confidencial", v * frac * fator, "Cotas de fundos (sigilo)", 0, 0
            elif tp in ("Ações", "Ações e outros TVM cedidos em empréstimo", "Cotas de Fundos",
                        "Brazilian Depository Receipt - BDR", "Investimento no Exterior"):
                yield "CONFIDENCIAL", "Confidencial", v * fator, f"{tp} (sigilo)", 0, 0

    linhas, resumo = [], []
    for cnpj, nome in PEERS.items():
        pl_f = plmap.get(cnpj)
        if not pl_f:
            continue
        exp = pd.DataFrame(list(explodir(cnpj, 1.0)),
                           columns=["ativo", "tipo", "valor", "desc", "aquis", "venda"])
        if exp.empty:
            exp = pd.DataFrame([("CONFIDENCIAL", "Confidencial", pl_f, "sem abertura", 0, 0)],
                               columns=exp.columns)
        exp = exp.groupby(["ativo", "tipo"], as_index=False).agg(
            valor=("valor", "sum"), desc=("desc", "first"), aquis=("aquis", "sum"), venda=("venda", "sum"))
        exp["cnpj"], exp["fundo"], exp["ym"], exp["pl"] = cnpj, nome, ym, pl_f
        exp["peso"] = exp["valor"] / pl_f
        linhas.append(exp)
        acoes = exp[exp["tipo"].isin(TIPOS_ACAO_LONG)]
        resumo.append({
            "cnpj": cnpj, "fundo": nome, "ym": ym, "pl": pl_f,
            "pct_acoes": acoes["valor"].sum() / pl_f,
            "pct_short": exp.loc[exp["tipo"] == "Vendido (short)", "valor"].sum() / pl_f,
            "pct_offshore": exp.loc[exp["tipo"] == "Offshore", "valor"].sum() / pl_f,
            "pct_opcoes": exp.loc[exp["tipo"].str.startswith("Opção"), "valor"].sum() / pl_f,
            "pct_caixa": exp.loc[exp["tipo"] == "Caixa/RF", "valor"].sum() / pl_f,
            "pct_confid": exp.loc[exp["tipo"] == "Confidencial", "valor"].sum() / pl_f,
            "n_acoes": int((acoes["peso"].abs() > 0.001).sum()),
            "aquis": exp["aquis"].sum(), "venda": exp["venda"].sum(),
        })
    return pd.concat(linhas, ignore_index=True), pd.DataFrame(resumo)


def coletar_cda(forcar: bool):
    hoje = date.today()
    todos, resumos, precos_cda = [], [], []
    for ym in meses(INICIO, hoje):
        # CVM republica o CDA conforme o sigilo vence: rebaixa os últimos 8 meses
        # se o arquivo local tiver mais de 5 dias
        p = RAW / f"cda_fi_{ym}.zip"
        velho = p.exists() and (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days >= 5
        recente = ym >= (hoje - timedelta(days=245)).strftime("%Y%m")
        p = baixar(URL_CDA.format(ym=ym), p, forcar or (recente and velho))
        if p is None:
            print(f"  CDA {ym}: ainda não publicado")
            continue
        pl, b2, b4, b7, conf, px = processar_cda_mes(p)
        exp, res = montar_exposicoes(ym, pl, b2, b4, b7, conf)
        precos_cda.append(px.assign(ym=ym).reset_index().rename(columns={"CD_ATIVO": "ativo"}))
        todos.append(exp)
        resumos.append(res)
        print(f"  CDA {ym}: {res['fundo'].nunique()} fundos | confid médio {res['pct_confid'].mean():.0%}")
    # 2ª passada: meses antigos cujo vínculo FIC->master só foi aprendido depois
    for i, r in enumerate(resumos):
        if (r["pct_confid"] > 0.5).any():
            ym = r["ym"].iloc[0]
            pl, b2, b4, b7, conf, _ = processar_cda_mes(RAW / f"cda_fi_{ym}.zip")
            todos[i], resumos[i] = montar_exposicoes(ym, pl, b2, b4, b7, conf)

    pos = pd.concat(todos, ignore_index=True)
    res = pd.concat(resumos, ignore_index=True)
    pos["data"] = pd.to_datetime(pos["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    res["data"] = pd.to_datetime(res["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    pos.to_parquet(DATA / "posicoes.parquet", index=False)
    res.to_parquet(DATA / "resumo_cda.parquet", index=False)
    pc = pd.concat(precos_cda, ignore_index=True)
    pc = pc[pc["count"] >= 2]          # exige ao menos 2 fundos para validar o preço
    pc.to_parquet(DATA / "precos_cda.parquet", index=False)
    print(f"-> posicoes.parquet: {len(pos)} linhas | meses {pos['ym'].min()}..{pos['ym'].max()}")
    return pos, res


# ─────────────────────────────────────────────────────────────────────────────
# 3) Preços (Yahoo chart API, ajustado por proventos) + CDI (BCB)
#    Fallback no dashboard: preço implícito do CDA (VL/QT), sem proventos.
# ─────────────────────────────────────────────────────────────────────────────
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{s}?period1={p1}&period2={p2}&interval=1d&events=div,split"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _yahoo(simbolo: str, p1: int, p2: int) -> pd.Series | None:
    import time
    for tentativa in range(3):
        try:
            r = requests.get(YAHOO.format(s=simbolo, p1=p1, p2=p2), headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (tentativa + 1))
                continue
            if r.status_code != 200:
                return None
            res = r.json()["chart"]["result"][0]
            adj = res["indicators"]["adjclose"][0]["adjclose"]
            idx = pd.to_datetime(res["timestamp"], unit="s") - pd.Timedelta(hours=3)
            return pd.Series(adj, index=idx.normalize(), name=simbolo).dropna()
        except Exception:
            time.sleep(2)
    return None


def coletar_precos(pos: pd.DataFrame):
    from concurrent.futures import ThreadPoolExecutor
    relevantes = pos[pos["tipo"].isin(["Ação", "Ação (emprestada)", "ETF", "BDR", "Recibo", "Vendido (short)"])
                     & (pos["peso"].abs() > 0.0005)]
    tks = sorted(t for t in relevantes["ativo"].unique() if re.fullmatch(r"[A-Z]{4}\d{1,2}", t))
    p1 = int(pd.Timestamp(INICIO - timedelta(days=45)).timestamp())
    p2 = int(pd.Timestamp(date.today() + timedelta(days=1)).timestamp())
    simbolos = {t + ".SA": t for t in tks} | {"^BVSP": "IBOV", "SMAL11.SA": "SMAL11", "BOVA11.SA": "BOVA11"}
    print(f"  Yahoo: {len(simbolos)} ativos...")
    with ThreadPoolExecutor(6) as ex:
        series = list(ex.map(lambda s: _yahoo(s, p1, p2), simbolos))
    ok = [s.rename(simbolos[s.name]) for s in series if s is not None and len(s)]
    px = pd.concat(ok, axis=1).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px.to_parquet(DATA / "precos_diarios.parquet")
    print(f"-> precos_diarios: {px.shape[1]}/{len(simbolos)} ativos, até {px.index.max().date()}")

    coletar_cdi()


def coletar_cdi():
    import time
    fim = date.today()
    for tentativa in range(4):
        try:
            r = requests.get(URL_CDI.format(ini=INICIO.strftime("%d/%m/%Y"), fim=fim.strftime("%d/%m/%Y")),
                             headers=UA, timeout=60)
            cdi = pd.DataFrame(r.json())
            break
        except Exception as e:
            print(f"   BCB tentativa {tentativa + 1}: {type(e).__name__}")
            time.sleep(10)
    else:
        print("   BCB indisponível — mantendo cdi.parquet anterior")
        return
    cdi["data"] = pd.to_datetime(cdi["data"], dayfirst=True)
    cdi["cdi"] = pd.to_numeric(cdi["valor"]) / 100
    cdi[["data", "cdi"]].to_parquet(DATA / "cdi.parquet", index=False)
    print(f"-> cdi: {len(cdi)} dias")


def gravar_nomes_ativos(pos: pd.DataFrame):
    nomes = (pos.sort_values("ym").groupby("ativo")["desc"].last().reset_index())
    nomes.to_parquet(DATA / "ativos.parquet", index=False)


if __name__ == "__main__":
    forcar = "--refresh" in sys.argv
    t0 = datetime.now()
    if "--so-precos" in sys.argv:
        pos = pd.read_parquet(DATA / "posicoes.parquet")
    else:
        print("[1/3] Informe diário")
        coletar_inf_diario(forcar)
        print("[2/3] CDA (carteiras)")
        pos, res = coletar_cda(forcar)
        gravar_nomes_ativos(pos)
    print("[3/3] Preços e benchmarks")
    coletar_precos(pos)
    print("[4/4] Backtest da projeção de carteiras")
    import projecao
    projecao.gerar_backtest()
    (DATA / "atualizado_em.txt").write_text(datetime.now().strftime("%d/%m/%Y %H:%M"))
    print(f"OK em {(datetime.now()-t0).seconds}s")
