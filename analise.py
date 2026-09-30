"""
Camada analítica do dashboard Peers Long Bias.
Lê os arquivos gerados por coletor_cvm.py (pasta data/) e entrega DataFrames prontos.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
AWR = "AWR Long Bias"
INICIO = pd.Timestamp("2025-04-10")

# trocas de ticker 1:1 (peso do mês anterior segue para o novo código)
RENOMES = {"ELET3": "AXIA3", "ELET6": "AXIA6", "EMBR3": "EMBJ3"}

TIPOS_LONG = ["Ação", "Ação (emprestada)", "BDR", "Recibo", "ETF", "Termo"]
TIPOS_EQUITY = TIPOS_LONG + ["Vendido (short)"]
LIMITE_CONFID = 0.25      # mês com mais de 25% do PL em sigilo não entra na atribuição


def _ler(nome):
    p = DATA / nome
    return pd.read_parquet(p) if p.exists() else None


class Base:
    def __init__(self):
        self.diario = _ler("diario.parquet")
        self.pos = _ler("posicoes.parquet")
        self.resumo = _ler("resumo_cda.parquet")
        self.px_cda = _ler("precos_cda.parquet")
        self.px_d = _ler("precos_diarios.parquet")
        self.cdi = _ler("cdi.parquet")
        setores = pd.read_csv(DATA / "setores.csv")
        self.setor_map = dict(zip(setores["ativo"].str[:4], setores["setor"].replace(
            {"Consumo Não Cíclico": "Consumo não Cíclico", "Comunicações": "Telecomunicações"})))
        try:
            self.atualizado = (DATA / "atualizado_em.txt").read_text().strip()
        except OSError:
            self.atualizado = "-"

        self.pos["ativo"] = self.pos["ativo"].replace(RENOMES)
        self.fundos = list(self.diario.groupby("fundo")["pl"].last().sort_values(ascending=False).index)
        self._cotas()
        self._retornos_ativos()
        self._equity()
        self._atribuicao()

    # ── cotas e benchmarks ────────────────────────────────────────────────────
    def _cotas(self):
        d = self.diario.pivot_table(index="data", columns="fundo", values="cota").sort_index()
        d = d[d.index >= INICIO].ffill()
        self.cotas = d / d.bfill().iloc[0]           # base 1 na 1ª data de cada fundo
        cdi = self.cdi.set_index("data")["cdi"]
        cdi = cdi[cdi.index > d.index[0]]
        self.cdi_idx = (1 + cdi).cumprod().reindex(d.index).ffill().fillna(1.0)
        if self.px_d is not None and "IBOV" in self.px_d:
            ib = self.px_d["IBOV"].dropna()
            ib = ib[ib.index >= d.index[0]]
            self.ibov_idx = (ib / ib.iloc[0]).reindex(d.index).ffill()
        else:
            self.ibov_idx = None
        m = self.cotas.resample("ME").last()
        base = self.cotas.iloc[[0]].copy()
        base.index = [self.cotas.index[0] - pd.Timedelta(days=1)]
        self.ret_mensal = pd.concat([base, m]).pct_change().iloc[1:]
        self.ret_mensal.index = self.ret_mensal.index.to_period("M").to_timestamp("M")
        ib_m = self.ibov_idx.resample("ME").last() if self.ibov_idx is not None else None
        self.ibov_mensal = pd.concat([pd.Series([1.0]), ib_m.reset_index(drop=True)]).pct_change().iloc[1:].values \
            if ib_m is not None else None
        cdi_m = self.cdi_idx.resample("ME").last()
        self.cdi_mensal = pd.Series(np.r_[cdi_m.iloc[0] - 1, cdi_m.pct_change().iloc[1:].values],
                                    index=self.ret_mensal.index)
        if ib_m is not None:
            self.ibov_mensal = pd.Series(self.ibov_mensal, index=self.ret_mensal.index)

    def metricas(self, ini=None, fim=None):
        c = self.cotas.copy()
        cdi, ib = self.cdi_idx.copy(), self.ibov_idx
        if ini is not None:
            c, cdi = c[c.index >= ini], cdi[cdi.index >= ini]
            ib = ib[ib.index >= ini] if ib is not None else None
        if fim is not None:
            c, cdi = c[c.index <= fim], cdi[cdi.index <= fim]
            ib = ib[ib.index <= fim] if ib is not None else None
        r = c.pct_change()
        rcdi = cdi.pct_change()
        anos = max((c.index[-1] - c.index[0]).days / 365.25, 1 / 12)
        out = pd.DataFrame(index=c.columns)
        out["Retorno"] = c.ffill().iloc[-1] / c.bfill().iloc[0] - 1
        out["Retorno a.a."] = (1 + out["Retorno"]) ** (1 / anos) - 1
        cdi_tot = cdi.iloc[-1] / cdi.iloc[0] - 1
        out["% CDI"] = out["Retorno"] / cdi_tot if cdi_tot else np.nan
        if ib is not None and len(ib):
            ib_tot = ib.iloc[-1] / ib.iloc[0] - 1
            out["vs Ibov (p.p.)"] = out["Retorno"] - ib_tot
            rib = ib.pct_change()
            cov = r.apply(lambda s: s.cov(rib))
            out["Beta Ibov"] = cov / rib.var()
        out["Vol a.a."] = r.std() * np.sqrt(252)
        exc = r.sub(rcdi, axis=0)
        out["Sharpe"] = exc.mean() / r.std() * np.sqrt(252)
        dd = c / c.cummax() - 1
        out["Max DD"] = dd.min()
        rm = c.resample("ME").last().pct_change().dropna(how="all")
        out["% meses +"] = (rm > 0).sum() / rm.notna().sum()
        pl = self.diario.pivot_table(index="data", columns="fundo", values="pl").ffill()
        out["PL (R$ mi)"] = pl.iloc[-1] / 1e6
        fl = self.diario.copy()
        if ini is not None:
            fl = fl[fl["data"] >= ini]
        if fim is not None:
            fl = fl[fl["data"] <= fim]
        out["Capt. líq. (R$ mi)"] = (fl.groupby("fundo")["captacao"].sum() - fl.groupby("fundo")["resgate"].sum()) / 1e6
        return out.sort_values("Retorno", ascending=False)

    def fluxos_mensais(self):
        d = self.diario.copy()
        d["mes"] = d["data"].dt.to_period("M").dt.to_timestamp("M")
        g = d.groupby(["fundo", "mes"]).agg(captacao=("captacao", "sum"), resgate=("resgate", "sum"))
        g["liquida"] = g["captacao"] - g["resgate"]
        pl = d.sort_values("data").groupby(["fundo", "mes"])["pl"].last()
        cot = d.sort_values("data").groupby(["fundo", "mes"])["cotistas"].last()
        return g.join(pl).join(cot).reset_index()

    # ── retornos mensais dos ativos ───────────────────────────────────────────
    def _retornos_ativos(self):
        meses = sorted(self.pos["data"].unique())
        idx = pd.DatetimeIndex(meses)
        # Yahoo (ajustado por proventos)
        ry = None
        if self.px_d is not None:
            m = self.px_d.resample("ME").last()
            m = m.reindex(idx.union(m.index)).loc[lambda x: x.index <= idx.max() + pd.offsets.MonthEnd(1)]
            ry = m.pct_change(fill_method=None)
        # CDA (VL/QT) com ajuste de desdobramento/grupamento
        w = self.px_cda.assign(data=pd.to_datetime(self.px_cda["ym"], format="%Y%m") + pd.offsets.MonthEnd(0))
        w["ativo"] = w["ativo"].replace(RENOMES)
        w = w.sort_values("count").drop_duplicates(["ativo", "data"], keep="last")
        w = w.pivot(index="data", columns="ativo", values="median").sort_index()
        razao = w / w.shift(1)
        r = razao - 1
        for k in [2, 3, 4, 5, 8, 10, 20, 100]:
            split = (np.abs(razao * k - 1) < 0.12) & (np.abs(r) > 0.35)     # desdobramento 1:k
            r = r.mask(split, razao * k - 1)
            grup = (np.abs(razao / k - 1) < 0.12) & (np.abs(r) > 0.35)      # grupamento k:1
            r = r.mask(grup, razao / k - 1)
        rc = r.clip(-0.7, 1.5)
        if ry is not None:
            ry = ry.reindex(index=rc.index.union(ry.index))
            comb = ry.combine_first(rc.reindex(ry.index))
        else:
            comb = rc
        self.ret_ativo = comb.sort_index()
        self.fonte_preco = "Yahoo (ajustado por proventos) + CDA/CVM (VL÷QT) como complemento"

    # ── exposição equity por ativo (longs + shorts consolidados por ticker) ──
    def _equity(self):
        p = self.pos[self.pos["tipo"].isin(TIPOS_EQUITY)].copy()
        eq = p.groupby(["fundo", "data", "ativo"], as_index=False).agg(
            peso=("peso", "sum"), valor=("valor", "sum"), desc=("desc", "first"),
            aquis=("aquis", "sum"), venda=("venda", "sum"))
        etf = eq["ativo"].str.startswith("ETF") | eq["ativo"].isin(["SMAL11", "BOVA11"])
        eq["setor"] = eq["ativo"].str[:4].map(self.setor_map)
        eq.loc[etf, "setor"] = "ETFs"
        bdr = eq["setor"].isna() & eq["ativo"].str.fullmatch(r"[A-Z0-9]{4}3[1-59]")
        eq.loc[bdr, "setor"] = "BDRs"
        eq["setor"] = eq["setor"].fillna("Outros")
        self.eq = eq
        r = self.resumo.copy()
        r["aberto"] = r["pct_confid"] <= LIMITE_CONFID
        self.meses_abertos = r[r["aberto"]].groupby("fundo")["data"].apply(sorted).to_dict()
        self.resumo_ok = r

    def ultimo_mes_aberto(self, fundo):
        m = self.meses_abertos.get(fundo, [])
        return m[-1] if m else None

    def carteira(self, fundo, data):
        return self.eq[(self.eq["fundo"] == fundo) & (self.eq["data"] == data)].sort_values("peso", ascending=False)

    # ── atribuição: contrib = peso(m-1) × retorno do ativo(m) ────────────────
    def _atribuicao(self):
        linhas = []
        for fundo, meses in self.meses_abertos.items():
            for d0 in meses:
                d1 = d0 + pd.offsets.MonthEnd(1)
                if d1 not in self.ret_ativo.index:
                    continue
                cart = self.eq[(self.eq["fundo"] == fundo) & (self.eq["data"] == d0)]
                rets = self.ret_ativo.loc[d1]
                c = cart[["ativo", "peso", "setor", "desc"]].copy()
                c["ret"] = c["ativo"].map(rets)
                c["contrib"] = c["peso"] * c["ret"]
                c["mes"], c["fundo"] = d1, fundo
                linhas.append(c)
        a = pd.concat(linhas, ignore_index=True)
        self.atrib = a
        # conciliação: explicado (ações) vs retorno da cota no mês
        exp = a.groupby(["fundo", "mes"]).agg(explicado=("contrib", "sum"),
                                              peso_sem_preco=("peso", lambda s: s[a.loc[s.index, "ret"].isna()].abs().sum()))
        rm = self.ret_mensal.stack().rename("retorno_cota")
        rm.index.names = ["mes", "fundo"]
        exp = exp.join(rm.swaplevel(), how="left")
        exp["residuo"] = exp["retorno_cota"] - exp["explicado"]
        self.concil = exp.reset_index()

    def contrib_acumulada(self, fundo=None, ini=None, fim=None):
        a = self.atrib if fundo is None else self.atrib[self.atrib["fundo"] == fundo]
        if ini is not None:
            a = a[a["mes"] >= ini]
        if fim is not None:
            a = a[a["mes"] <= fim]
        g = a.groupby(["fundo", "ativo"] if fundo is None else ["ativo"]).agg(
            contrib=("contrib", "sum"), peso_medio=("peso", "mean"), meses=("mes", "nunique"),
            ret_medio=("ret", "mean"), setor=("setor", "first"))
        return g.sort_values("contrib", ascending=False).reset_index()

    def contrib_setor(self, ini=None, fim=None):
        a = self.atrib
        if ini is not None:
            a = a[a["mes"] >= ini]
        if fim is not None:
            a = a[a["mes"] <= fim]
        return a.groupby(["fundo", "setor"])["contrib"].sum().unstack().fillna(0)

    # ── consenso / crowding ───────────────────────────────────────────────────
    def consenso(self, data):
        """Para cada ativo: nº de fundos comprados, peso médio, peso AWR, etc. (último mês aberto <= data)."""
        cortes = []
        for f in self.fundos:
            m = [x for x in self.meses_abertos.get(f, []) if x <= data]
            if m:
                cortes.append(self.carteira(f, m[-1]).assign(data_ref=m[-1]))
        c = pd.concat(cortes, ignore_index=True)
        c = c[c["ativo"].str.fullmatch(r"[A-Z]{4}\d{1,2}")]
        n_fundos = c["fundo"].nunique()
        long = c[c["peso"] > 0.002]
        short = c[c["peso"] < -0.002]
        g = long.groupby("ativo").agg(n_long=("fundo", "nunique"), peso_medio=("peso", "mean"),
                                      peso_max=("peso", "max"), setor=("setor", "first"))
        g["n_short"] = short.groupby("ativo")["fundo"].nunique()
        tot = c.pivot_table(index="ativo", columns="fundo", values="peso", aggfunc="sum").fillna(0)
        g["peso_medio_todos"] = tot.reindex(g.index).mean(axis=1)
        g["peso_awr"] = tot[AWR].reindex(g.index) if AWR in tot else np.nan
        g["n_short"] = g["n_short"].fillna(0).astype(int)
        g["peso_awr"] = g["peso_awr"].fillna(0)
        g["ativo_vs_peers"] = g["peso_awr"] - tot.drop(columns=[AWR], errors="ignore").reindex(g.index).mean(axis=1)
        return g.sort_values(["n_long", "peso_medio"], ascending=False).reset_index(), tot, n_fundos

    def similaridade(self, tot):
        v = tot.clip(lower=0)
        norm = np.sqrt((v ** 2).sum())
        sim = (v.T @ v) / np.outer(norm, norm)
        return pd.DataFrame(sim, index=v.columns, columns=v.columns)

    # ── movimentações ─────────────────────────────────────────────────────────
    def movimentos(self, fundo):
        meses = self.meses_abertos.get(fundo, [])
        e = self.eq[self.eq["fundo"] == fundo].pivot_table(index="ativo", columns="data", values="peso",
                                                              aggfunc="sum").fillna(0)
        out = []
        for a, b in zip(meses[:-1], meses[1:]):
            if a not in e or b not in e:
                continue
            # variação de peso descontando efeito preço (peso esperado se não houvesse trade)
            r = self.ret_ativo.loc[b] if b in self.ret_ativo.index else pd.Series(dtype=float)
            esperado = e[a] * (1 + r.reindex(e.index).fillna(0))
            esperado = esperado / (1 + (e[a] * r.reindex(e.index).fillna(0)).sum())
            delta = e[b] - esperado
            for at in e.index:
                pa, pb = e.at[at, a], e.at[at, b]
                if abs(pa) < 0.001 and abs(pb) < 0.001:
                    continue
                if abs(pa) < 0.001:
                    acao = "Nova posição"
                elif abs(pb) < 0.001:
                    acao = "Zerou"
                elif delta[at] > 0.005:
                    acao = "Aumentou"
                elif delta[at] < -0.005:
                    acao = "Reduziu"
                else:
                    continue
                out.append({"mes": b, "ativo": at, "acao": acao, "peso_antes": pa, "peso_depois": pb,
                            "delta_ativo": delta[at]})
        return pd.DataFrame(out)

    def concentracao(self):
        g = self.eq[self.eq["peso"] > 0].sort_values("peso", ascending=False).groupby(["fundo", "data"])
        out = pd.DataFrame({
            "top5": g["peso"].apply(lambda s: s.head(5).sum()),
            "top10": g["peso"].apply(lambda s: s.head(10).sum()),
            "hhi": g["peso"].apply(lambda s: ((s / s.sum()) ** 2).sum()),
            "n": g["peso"].apply(lambda s: int((s > 0.005).sum())),
        }).reset_index()
        out = out.merge(self.resumo_ok[["fundo", "data", "aberto", "pct_acoes", "pct_short",
                                        "pct_offshore", "pct_caixa", "pct_confid"]], on=["fundo", "data"])
        out["n_efetivo"] = 1 / out["hhi"]
        return out[out["aberto"]]


if __name__ == "__main__":
    b = Base()
    pd.set_option("display.width", 220)
    print(b.metricas().round(3).to_string())
    print(b.concil.groupby("fundo")[["explicado", "retorno_cota", "residuo"]].sum().round(3))
    print(b.concil.groupby("fundo")["peso_sem_preco"].mean().round(3))
    print(b.contrib_acumulada(AWR).head(10).round(4))
