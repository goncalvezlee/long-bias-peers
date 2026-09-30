"""
Projeção das carteiras em sigilo.

Parte da última carteira aberta na CVM (data-base T0) e usa a cota diária divulgada
depois de T0 para estimar a exposição atual:

1) Carteira congelada: pesos de T0 derivando só com o preço até hoje.
   Se o retorno dela cola na cota real, o gestor mexeu pouco.
2) Carteira calibrada: regressão ridge dos retornos diários da cota (em excesso ao CDI)
   contra as ações da carteira-base + Ibovespa (hedge/beta via futuro), com os pesos
   "puxados" para a carteira congelada. Só se afasta dela quando a cota exige.
3) Backtest: o mesmo método em janelas passadas onde a carteira seguinte é pública,
   para calibrar a força da âncora (lambda) e medir o acerto.
"""
import numpy as np
import pandas as pd

from analise import AWR

JANELA = 63              # pregões usados na calibração (≈ 3 meses mais recentes)
PESO_MIN = 0.003         # ativos abaixo disso entram na cesta "demais ações"
LAMBDAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
MODELOS = ["escala", "setor", "ativo"]
TIPOS_EQ = ["Ação", "Ação (emprestada)", "BDR", "Recibo", "ETF", "Termo", "Vendido (short)"]


class Projecao:
    def __init__(self, b, backtest=None):
        self.b = b
        self.px = b.px_d.sort_index().ffill(limit=5)
        self.cotas = b.diario.pivot_table(index="data", columns="fundo", values="cota").sort_index()
        self.cdi_d = b.cdi.set_index("data")["cdi"].sort_index()
        self.lam, self.modelo = 3e-4, "escala"
        self.backtest = self._backtest() if backtest is None else backtest
        if not self.backtest.empty:
            # score normalizado: erro de peso por ativo + erro de exposição agregada
            g = self.backtest.groupby(["modelo", "lambda"])[
                ["mae_cal", "mae_congelada", "erro_exp_cal", "erro_exp_congelada"]].mean()
            score = g["mae_cal"] / g["mae_congelada"] + g["erro_exp_cal"] / g["erro_exp_congelada"]
            self.modelo, self.lam = score.idxmin()
            self.lam = float(self.lam)
        self.resultados = {f: self.projetar(f) for f in b.fundos if b.meses_abertos.get(f)}

    # ── peças ─────────────────────────────────────────────────────────────────
    def _base(self, fundo, t0):
        c = self.b.carteira(fundo, t0)
        c = c[c["ativo"].isin(c["ativo"])]  # cópia
        res = self.b.resumo_ok
        r = res[(res["fundo"] == fundo) & (res["data"] == t0)]
        caixa = float(r["pct_caixa"].iloc[0]) if len(r) else 0.0
        p0 = self.px[self.px.index <= t0].iloc[-1] if (self.px.index <= t0).any() else None
        tem_px = c["ativo"].isin(self.px.columns) & c["ativo"].map(lambda a: p0 is not None and a in p0 and pd.notna(p0[a]))
        grandes = c[tem_px & (c["peso"].abs() >= PESO_MIN)]
        resto = c.loc[~c.index.isin(grandes.index), "peso"].sum()      # cesta proxy Ibovespa
        return grandes.set_index("ativo")["peso"], float(resto), caixa, c

    def _janela(self, fundo, t0, t1):
        cot = self.cotas[fundo].dropna()
        cot = cot[(cot.index >= cot[cot.index <= t0].index.max()) & (cot.index <= t1)]
        return cot

    def _congelada(self, w, resto, caixa, datas, t0):
        """Valor da carteira congelada (base 1 em T0) e pesos derivados por data."""
        p = self.px.reindex(self.px.index.union(datas)).ffill().reindex(datas)
        p0 = self.px[self.px.index <= t0].iloc[-1]
        rel = p[w.index] / p0[w.index]
        ib = (p["IBOV"] / p0["IBOV"]) if "IBOV" in p else 1.0
        cdi = (1 + self.cdi_d).cumprod()
        cdi = cdi.reindex(cdi.index.union(datas)).ffill().reindex(datas)
        c0 = (1 + self.cdi_d).cumprod()
        c0 = c0[c0.index <= t0].iloc[-1] if (c0.index <= t0).any() else 1.0
        outros = 1 - w.sum() - resto - caixa        # offshore, derivativos, ajustes: parado
        valores = rel * w
        V = valores.sum(axis=1) + resto * ib + caixa * (cdi / c0) + outros
        pesos = valores.div(V, axis=0)
        return V, pesos

    def _calibrar(self, fundo, w_prior, datas, lam, modelo="setor", setores=None):
        """Ridge ancorado: retorno em excesso da cota ~ Σ_g m_g · (contribuição do grupo g) + h · Ibov.

        modelo = "escala" (1 grupo = book inteiro), "setor" (1 grupo por setor) ou "ativo".
        m_g é puxado para 1 (carteira congelada) e h para 0 (sem hedge adicional).
        """
        cot = self.cotas[fundo].reindex(datas)
        p = self.px.reindex(self.px.index.union(datas)).ffill().reindex(datas)
        cdi = self.cdi_d.reindex(self.cdi_d.index.union(datas)).ffill().reindex(datas).fillna(0)
        y = (cot.pct_change() - cdi).iloc[1:]
        R = p[list(w_prior.index)].pct_change().iloc[1:].sub(cdi.iloc[1:], axis=0)
        rib = (p["IBOV"].pct_change().iloc[1:] - cdi.iloc[1:])
        ok = y.notna() & R.notna().all(axis=1) & rib.notna()
        y, R, rib = y[ok], R[ok], rib[ok]
        if len(y) < 15:
            return None
        if modelo == "escala":
            grupos = pd.Series("book", index=w_prior.index)
        elif modelo == "ativo":
            grupos = pd.Series(w_prior.index, index=w_prior.index)
        else:
            grupos = (setores.reindex(w_prior.index).fillna("Outros") if setores is not None
                      else pd.Series("book", index=w_prior.index))
        contrib = R * w_prior                       # contribuição diária de cada ativo
        G = contrib.T.groupby(grupos).sum().T       # contribuição por grupo
        X = np.c_[G.values, rib.values]
        yv = y.values
        # âncora: m_g -> 1 ; h -> 0. Penalidade normalizada pelo tamanho do grupo.
        tam = np.abs(w_prior.groupby(grupos).sum().reindex(G.columns).values) + 0.02
        d = np.r_[lam * tam, lam * 4.0]
        prior = np.r_[np.ones(G.shape[1]), 0.0]
        sol = np.linalg.solve(X.T @ X + np.diag(d), X.T @ yv + d * prior)
        m = pd.Series(np.clip(sol[:-1], 0, 3), index=G.columns)
        fit = X @ np.r_[m.values, sol[-1]]
        sst = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - ((yv - fit) ** 2).sum() / sst
        r2_prior = 1 - ((yv - X @ prior) ** 2).sum() / sst
        te = (yv - fit).std() * np.sqrt(252)
        w = w_prior * grupos.map(m).values
        return {"w": w, "mult": m, "hedge": float(sol[-1]), "r2": r2, "r2_congelada": r2_prior,
                "te": te, "n": len(yv), "modelo": modelo}

    # ── projeção até hoje ─────────────────────────────────────────────────────
    def projetar(self, fundo, t0=None, t1=None, lam=None, modelo=None):
        lam = self.lam if lam is None else lam
        modelo = self.modelo if modelo is None else modelo
        meses = self.b.meses_abertos.get(fundo, [])
        if not meses:
            return None
        t0 = t0 or meses[-1]
        cot = self.cotas[fundo].dropna()
        t1 = t1 or cot.index.max()
        w, resto, caixa, cart0 = self._base(fundo, t0)
        cj = self._janela(fundo, t0, t1)
        if len(cj) < 10 or w.empty:
            return None
        datas = cj.index
        V, pesos_d = self._congelada(w, resto, caixa, datas, t0)
        real = cj / cj.iloc[0]
        cong = V / V.iloc[0]
        rj = real.pct_change().dropna()
        rc = cong.pct_change().reindex(rj.index)
        te_cong = (rj - rc).std() * np.sqrt(252)
        corr = rj.corr(rc)

        # calibração nos últimos JANELA pregões, ancorada nos pesos derivados do início da janela
        dj = datas[-(JANELA + 1):]
        prior = pesos_d.loc[dj[0]]
        cal = self._calibrar(fundo, prior, dj, lam, modelo, cart0.set_index("ativo")["setor"])

        # pesos de hoje
        w_hoje = pesos_d.iloc[-1]
        out = pd.DataFrame({"peso_base": w, "peso_congelado": w_hoje})
        if cal is not None:
            # a calibração estima os pesos médios da janela; deriva do meio da janela até hoje
            p = self.px.reindex(self.px.index.union(datas)).ffill().reindex(datas)
            meio = dj[len(dj) // 2]
            rel = p.loc[datas[-1], w.index] / p.loc[meio, w.index]
            est = cal["w"] * rel
            out["peso_estimado"] = est / (1 + (cal["w"] * (rel - 1)).sum())
            out["mult_grupo"] = (est / out["peso_congelado"]).where(out["peso_congelado"].abs() > 1e-6)
        else:
            out["peso_estimado"] = out["peso_congelado"]
        p0 = self.px[self.px.index <= t0].iloc[-1]
        pf = self.px.reindex(self.px.index.union([datas[-1]])).ffill().loc[datas[-1]]
        out["ret_desde_base"] = pf[w.index] / p0[w.index] - 1
        out["contrib_desde_base"] = out["peso_base"] * out["ret_desde_base"]
        out["delta_vs_congelado"] = out["peso_estimado"] - out["peso_congelado"]
        setor = cart0.set_index("ativo")["setor"]
        out["setor"] = setor.reindex(out.index)
        out = out.sort_values("peso_estimado", ascending=False)

        long_est = out["peso_estimado"].sum() + resto
        return {
            "fundo": fundo, "t0": t0, "t1": datas[-1], "dias": len(datas) - 1,
            "real": real, "congelada": cong, "ret_real": real.iloc[-1] - 1, "ret_congelada": cong.iloc[-1] - 1,
            "te_congelada": te_cong, "corr": corr, "pesos": out, "resto": resto, "caixa": caixa,
            "cal": cal, "hedge": cal["hedge"] if cal else 0.0,
            "exp_base": w.sum() + resto, "exp_congelada": pesos_d.iloc[-1].sum() + resto,
            "exp_estimada": long_est + (cal["hedge"] if cal else 0.0),
        }

    # ── backtest: base m → projeção m+k, comparada com a carteira real de m+k ──
    def _backtest(self):
        linhas = []
        for f, meses in self.b.meses_abertos.items():
            for i, t0 in enumerate(meses):
                for t1 in meses[i + 1:]:
                    k = (t1.year - t0.year) * 12 + t1.month - t0.month
                    if k not in (3, 6):
                        continue
                    real = self.b.carteira(f, t1).set_index("ativo")["peso"]
                    for modelo, lam in [(mo, la) for mo in MODELOS for la in LAMBDAS]:
                        r = self.projetar(f, t0, t1, lam, modelo)
                        if r is None or r["cal"] is None:
                            continue
                        p = r["pesos"]
                        alvo = real.reindex(p.index).fillna(0)
                        mudanca_real = alvo - p["peso_congelado"]
                        mudanca_est = p["peso_estimado"] - p["peso_congelado"]
                        relevante = mudanca_real.abs() > 0.01
                        acerto = (np.sign(mudanca_real[relevante]) == np.sign(mudanca_est[relevante])).mean() \
                            if relevante.any() else np.nan
                        exp_real = real[real.index.isin(p.index)].sum()
                        linhas.append({
                            "fundo": f, "t0": t0, "t1": t1, "meses": k, "lambda": lam, "modelo": modelo,
                            "erro_exp_congelada": abs(p["peso_congelado"].sum() - exp_real),
                            "erro_exp_cal": abs(p["peso_estimado"].sum() - exp_real),
                            "mae_congelada": (alvo - p["peso_congelado"]).abs().mean(),
                            "mae_cal": (alvo - p["peso_estimado"]).abs().mean(),
                            "acerto_direcao": acerto,
                            "novos": real[~real.index.isin(p.index) & (real > 0)].sum(),
                            "zerados": (alvo.abs() < 0.001).sum() / max(len(alvo), 1),
                        })
        return pd.DataFrame(linhas)

    def erro_por_fundo(self):
        bt = self.backtest[(self.backtest["lambda"] == self.lam) & (self.backtest["modelo"] == self.modelo)]
        return bt.groupby("fundo").agg(erro_peso=("mae_cal", "mean"), erro_exp=("erro_exp_cal", "mean"),
                                       novos=("novos", "mean"), janelas=("t0", "size"))

    def resumo_backtest(self):
        bt = self.backtest[(self.backtest["lambda"] == self.lam) & (self.backtest["modelo"] == self.modelo)]
        if bt.empty:
            return bt
        g = bt.groupby("meses").agg(janelas=("fundo", "size"), mae_congelada=("mae_congelada", "mean"),
                                    mae_cal=("mae_cal", "mean"), acerto=("acerto_direcao", "mean"),
                                    erro_exp_congelada=("erro_exp_congelada", "mean"),
                                    erro_exp_cal=("erro_exp_cal", "mean"),
                                    novos=("novos", "mean"))
        g["ganho"] = 1 - g["mae_cal"] / g["mae_congelada"]
        return g.reset_index()

    # ── visão consolidada ─────────────────────────────────────────────────────
    def tabela_fundos(self):
        linhas = []
        for f, r in self.resultados.items():
            if r is None:
                continue
            gap = r["ret_real"] - r["ret_congelada"]
            te = r["te_congelada"]
            conf = "Alta" if te < 0.04 and abs(gap) < 0.04 else ("Média" if te < 0.08 and abs(gap) < 0.10 else "Baixa")
            linhas.append({
                "Fundo": f, "Base": r["t0"], "Pregões": r["dias"], "Cota real": r["ret_real"],
                "Carteira congelada": r["ret_congelada"], "Diferença": gap, "TE congelada": te,
                "Correlação": r["corr"], "R² calibrado": r["cal"]["r2"] if r["cal"] else np.nan,
                "Exposição base": r["exp_base"], "Exposição estimada": r["exp_estimada"],
                "Hedge Ibov est.": r["hedge"], "Aderência": conf,
            })
        return pd.DataFrame(linhas)

    def consenso_projetado(self):
        linhas = []
        for f, r in self.resultados.items():
            if r is None:
                continue
            p = r["pesos"]
            linhas.append(pd.DataFrame({"fundo": f, "ativo": p.index, "peso": p["peso_estimado"].values,
                                        "peso_base": p["peso_base"].values,
                                        "peso_congelado": p["peso_congelado"].values, "setor": p["setor"].values}))
        c = pd.concat(linhas, ignore_index=True)
        tot = c.pivot_table(index="ativo", columns="fundo", values="peso", aggfunc="sum").fillna(0)
        base = c.pivot_table(index="ativo", columns="fundo", values="peso_base", aggfunc="sum").fillna(0)
        peers = [x for x in tot.columns if x != AWR]
        cong = c.pivot_table(index="ativo", columns="fundo", values="peso_congelado", aggfunc="sum").fillna(0)
        g = pd.DataFrame({
            "n_fundos": (tot[peers] > 0.005).sum(axis=1),
            "peso_medio_peers": tot[peers].mean(axis=1),
            "peso_medio_base": base[peers].mean(axis=1),
            "peso_medio_congelado": cong[peers].mean(axis=1),
            "peso_awr": tot[AWR] if AWR in tot else 0.0,
            "setor": c.groupby("ativo")["setor"].first(),
        })
        g["var_media"] = g["peso_medio_peers"] - g["peso_medio_base"]
        return g.sort_values("peso_medio_peers", ascending=False).reset_index(), tot


def gerar_backtest():
    """Roda o backtest completo (lento) e grava em data/ para o app."""
    from analise import DATA, Base
    pj = Projecao(Base())
    pj.backtest.to_parquet(DATA / "projecao_backtest.parquet", index=False)
    print(f"-> projecao_backtest.parquet: {len(pj.backtest)} linhas | modelo {pj.modelo}, lambda {pj.lam}")
    return pj


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pj = gerar_backtest()
    print(pj.resumo_backtest().round(4).to_string())
    print(pj.tabela_fundos().to_string())
