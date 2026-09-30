"""
Peers Long Bias — Raio-X das carteiras (dados públicos CVM)
Rodar local:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analise import AWR, DATA, Base
from projecao import Projecao

st.set_page_config(page_title="Long Bias — Raio-X dos Peers", page_icon="📊", layout="wide")

# AWR mostra carteira só até o mesmo mês dos peers (evita expor posições mais novas)
MESMO_CORTE_AWR = os.getenv("MESMO_CORTE_AWR", "1") == "1"

# ── paleta (validada p/ fundo escuro: dataviz validate_palette, modo dark, all-pairs) ──
GOLD, BLUE, AQUA = "#c98500", "#3987e5", "#199e70"
GRAY, GRAY_L = "#5f5e5a", "#8a8984"
INK, INK2, INK3 = "#ffffff", "#c3c2b7", "#8a8984"
SURF, PLANE, GRID = "#1a1a19", "#0d0d0d", "#2c2c2a"
NEG, MID, POS = "#e66767", "#383835", "#3987e5"
DIVERG = [[0, NEG], [0.5, MID], [1, POS]]

st.markdown(f"""
<style>
.stApp {{ background:{PLANE}; }}
.block-container {{ padding-top:1.6rem; max-width:1500px; }}
h1, h2, h3 {{ letter-spacing:-0.01em; }}
.kpi {{ background:{SURF}; border:1px solid {GRID}; border-radius:10px; padding:14px 16px; }}
.kpi .l {{ color:{INK3}; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }}
.kpi .v {{ color:{INK}; font-size:1.55rem; font-weight:600; margin-top:2px; }}
.kpi .s {{ color:{INK2}; font-size:.8rem; }}
.nota {{ color:{INK3}; font-size:.8rem; }}
.tag-awr {{ color:{GOLD}; font-weight:600; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(ttl=6 * 3600, show_spinner="Carregando dados da CVM…")
def carregar():
    b = Base()
    if MESMO_CORTE_AWR:
        ult = sorted(m[-1] for f, m in b.meses_abertos.items() if f != AWR and m)
        corte = ult[len(ult) // 2] if ult else None       # corte típico (mediana) dos peers
        if corte is not None and AWR in b.meses_abertos:
            b.meses_abertos[AWR] = [m for m in b.meses_abertos[AWR] if m <= corte]
            b.atrib = b.atrib[(b.atrib["fundo"] != AWR) | (b.atrib["mes"] <= corte + pd.offsets.MonthEnd(1))]
            b.concil = b.concil[(b.concil["fundo"] != AWR) | (b.concil["mes"] <= corte + pd.offsets.MonthEnd(1))]
            b.eq = b.eq[(b.eq["fundo"] != AWR) | (b.eq["data"] <= corte)]
            b.resumo_ok = b.resumo_ok[(b.resumo_ok["fundo"] != AWR) | (b.resumo_ok["data"] <= corte)]
    return b


b = carregar()


@st.cache_resource(ttl=6 * 3600, show_spinner="Projetando carteiras…")
def carregar_projecao():
    bt = DATA / "projecao_backtest.parquet"
    return Projecao(b, pd.read_parquet(bt) if bt.exists() else None)


pj = carregar_projecao()
FUNDOS = b.fundos
PEERS = [f for f in FUNDOS if f != AWR]


# ── helpers de formatação e gráficos ──────────────────────────────────────────
def pct(x, d=1, sinal=False):
    if x is None or pd.isna(x):
        return "–"
    s = f"{x*100:+.{d}f}%" if sinal else f"{x*100:.{d}f}%"
    return s.replace(".", ",")


def num(x, d=1):
    return "–" if pd.isna(x) else f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mes_label(d):
    meses = "jan fev mar abr mai jun jul ago set out nov dez".split()
    return f"{meses[d.month-1]}/{str(d.year)[2:]}"


def layout(fig, h=420, legenda=True, **kw):
    fig.update_layout(
        height=h, paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=INK2, size=12),
        margin=dict(l=10, r=10, t=36, b=10), hoverlabel=dict(bgcolor="#262624", font_color=INK),
        showlegend=legenda, legend=dict(orientation="h", y=1.08, x=0, font=dict(color=INK2)),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK3))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#4a4a47", linecolor=GRID, tickfont=dict(color=INK3))
    return fig


def cor_fundo(f, destaques):
    if f == AWR:
        return GOLD
    if destaques and f == destaques[0]:
        return BLUE
    if len(destaques) > 1 and f == destaques[1]:
        return AQUA
    return GRAY


def kpi(col, rotulo, valor, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{rotulo}</div><div class="v">{valor}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def tabela(df, formatos, altura=None, destaque_awr=True):
    sty = df.style.format(formatos, na_rep="–")
    if destaque_awr and "Fundo" in df.columns:
        sty = sty.apply(lambda r: [f"color:{GOLD};font-weight:600" if r["Fundo"] == AWR else "" for _ in r], axis=1)
    st.dataframe(sty, hide_index=True, width="stretch", height=altura)


# ── cabeçalho e filtros globais ───────────────────────────────────────────────
st.markdown(f"## Long Bias · Raio-X dos Peers <span class='nota'>&nbsp;dados públicos CVM · "
            f"atualizado em {b.atualizado}</span>", unsafe_allow_html=True)

hoje = b.cotas.index.max()
periodos = {
    "Desde o início do AWR (10/04/25)": b.cotas.index.min(),
    "2026 (YTD)": pd.Timestamp("2025-12-31"),
    "Últimos 12 meses": hoje - pd.DateOffset(years=1),
    "Últimos 6 meses": hoje - pd.DateOffset(months=6),
    "2025": b.cotas.index.min(),
}
f1, f2, f3 = st.columns([1.3, 2.4, 1])
per = f1.selectbox("Período", list(periodos), index=0)
ini = periodos[per]
fim = pd.Timestamp("2025-12-31") if per == "2025" else hoje
destaques = f2.multiselect("Destacar peers (até 2)", PEERS, default=[], max_selections=2,
                           placeholder="AWR sempre em dourado · escolha até 2 peers para comparar")
f3.markdown(f"<div class='nota' style='margin-top:30px'>Cotas até {hoje:%d/%m/%Y}</div>", unsafe_allow_html=True)

abas = st.tabs(["Visão geral", "O que carregou cada fundo", "Carteiras", "Projeção (hoje)",
                "Consenso & AWR vs peers", "Movimentações", "Captação & PL", "Metodologia"])

# ═════════════════════════════════════════════════════════════════════════════
# 1) VISÃO GERAL
# ═════════════════════════════════════════════════════════════════════════════
with abas[0]:
    met = b.metricas(ini, fim)
    rk = list(met.index)
    a = met.loc[AWR]
    c = st.columns(5)
    kpi(c[0], "AWR · retorno no período", pct(a["Retorno"], 1, True), f"{rk.index(AWR)+1}º de {len(rk)} no grupo")
    kpi(c[1], "Mediana dos peers", pct(met.drop(AWR)["Retorno"].median(), 1, True),
        f"AWR {pct(a['Retorno'] - met.drop(AWR)['Retorno'].median(), 1, True).replace('%', ' p.p.')}")
    kpi(c[2], "AWR · volatilidade a.a.", pct(a["Vol a.a."]),
        f"menor vol: {met['Vol a.a.'].idxmin()}" if met["Vol a.a."].idxmin() != AWR else "a menor do grupo")
    kpi(c[3], "AWR · Sharpe", num(a["Sharpe"], 2), f"{sorted(met['Sharpe'], reverse=True).index(a['Sharpe'])+1}º no grupo")
    kpi(c[4], "AWR · max drawdown", pct(a["Max DD"]),
        f"{sorted(met['Max DD'], reverse=True).index(a['Max DD'])+1}º menor queda")

    st.markdown("#### Retorno acumulado")
    cot = b.cotas[(b.cotas.index >= ini) & (b.cotas.index <= fim)]
    cot = cot / cot.bfill().iloc[0]
    fig = go.Figure()
    ordem = [f for f in FUNDOS if f != AWR and f not in destaques] + destaques + [AWR]
    for f in ordem:
        s = cot[f].dropna()
        if s.empty:
            continue
        cor = cor_fundo(f, destaques)
        forte = f == AWR or f in destaques
        fig.add_trace(go.Scatter(
            x=s.index, y=s - 1, name=f, mode="lines",
            line=dict(color=cor, width=2.4 if forte else 1.1), opacity=1 if forte else 0.55,
            hovertemplate=f"<b>{f}</b><br>%{{x|%d/%m/%y}}: %{{y:.1%}}<extra></extra>",
            showlegend=forte))
        if forte:
            fig.add_annotation(x=s.index[-1], y=s.iloc[-1] - 1, text=f" {f} {pct(s.iloc[-1]-1,1,True)}",
                               showarrow=False, xanchor="left", font=dict(color=INK, size=11))
    cdi = b.cdi_idx[(b.cdi_idx.index >= ini) & (b.cdi_idx.index <= fim)]
    cdi = cdi / cdi.iloc[0]
    fig.add_trace(go.Scatter(x=cdi.index, y=cdi - 1, name="CDI", line=dict(color=INK3, width=1.5, dash="dash"),
                             hovertemplate="CDI %{x|%d/%m/%y}: %{y:.1%}<extra></extra>"))
    if b.ibov_idx is not None:
        ib = b.ibov_idx[(b.ibov_idx.index >= ini) & (b.ibov_idx.index <= fim)].dropna()
        ib = ib / ib.iloc[0]
        fig.add_trace(go.Scatter(x=ib.index, y=ib - 1, name="Ibovespa", line=dict(color=INK2, width=1.5, dash="dot"),
                                 hovertemplate="Ibovespa %{x|%d/%m/%y}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="demais peers", mode="lines", line=dict(color=GRAY, width=1.1)))
    layout(fig, 470, hovermode="closest", yaxis=dict(tickformat=".0%"), xaxis=dict(range=[cot.index[0], cot.index[-1] + pd.Timedelta(days=45)]))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Ranking do período")
    t = met.reset_index().rename(columns={"fundo": "Fundo", "index": "Fundo"})
    t.insert(0, "#", range(1, len(t) + 1))
    fmt = {c: "{:.1%}" for c in ["Retorno", "Retorno a.a.", "vs Ibov (p.p.)", "Vol a.a.", "Max DD", "% meses +"]}
    fmt.update({"% CDI": "{:.0%}", "Beta Ibov": "{:.2f}", "Sharpe": "{:.2f}", "PL (R$ mi)": "{:,.0f}",
                "Capt. líq. (R$ mi)": "{:+,.0f}"})
    tabela(t, fmt, altura=565)

    st.markdown("#### Risco × retorno")
    fig = go.Figure()
    for f in met.index:
        cor = cor_fundo(f, destaques)
        forte = f == AWR or f in destaques
        fig.add_trace(go.Scatter(
            x=[met.at[f, "Vol a.a."]], y=[met.at[f, "Retorno"]], mode="markers+text", name=f,
            marker=dict(size=14 if forte else 10, color=cor, line=dict(color=SURF, width=2)),
            text=[f], textposition="top center", textfont=dict(color=INK if forte else INK3, size=11),
            hovertemplate=f"<b>{f}</b><br>Vol %{{x:.1%}} · Retorno %{{y:.1%}}<br>Sharpe {met.at[f,'Sharpe']:.2f}<extra></extra>",
            showlegend=False))
    layout(fig, 430, xaxis=dict(title="Volatilidade anualizada", tickformat=".0%"),
           yaxis=dict(title="Retorno no período", tickformat=".0%"))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Retorno mês a mês")
    rm = b.ret_mensal[(b.ret_mensal.index >= ini.to_period("M").to_timestamp("M")) & (b.ret_mensal.index <= fim)]
    rm = rm[met.index]
    extra = pd.DataFrame({"CDI": b.cdi_mensal.reindex(rm.index)})
    if b.ibov_idx is not None:
        extra["Ibovespa"] = b.ibov_mensal.reindex(rm.index)
    rm = pd.concat([rm, extra], axis=1).T
    z = rm.values
    lim = np.nanpercentile(np.abs(z), 95)
    fig = go.Figure(go.Heatmap(
        z=z, x=[mes_label(d) for d in rm.columns], y=rm.index, colorscale=DIVERG, zmid=0, zmin=-lim, zmax=lim,
        text=[[pct(v, 1) for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=10, color=INK),
        xgap=2, ygap=2, hovertemplate="%{y} · %{x}: %{text}<extra></extra>", colorbar=dict(tickformat=".0%")))
    layout(fig, 60 + 28 * len(rm), legenda=False, yaxis=dict(autorange="reversed"))
    fig.update_yaxes(tickfont=dict(color=INK2))
    st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# 2) ATRIBUIÇÃO
# ═════════════════════════════════════════════════════════════════════════════
with abas[1]:
    st.markdown("Contribuição de cada ação = **peso no fim do mês anterior × retorno da ação no mês** "
                "(carteira aberta via look-through FIC → master). Soma em p.p. do PL do fundo.")
    ca = b.contrib_acumulada(None, ini, fim)
    ca = ca[ca["ativo"].str.fullmatch(r"[A-Z]{4}\d{1,2}|SMAL11|BOVA11")]
    meses_cob = b.atrib[(b.atrib["mes"] >= ini) & (b.atrib["mes"] <= fim)]
    if meses_cob.empty:
        st.info("Sem carteiras abertas nesse período (sigilo CVM de até 6 meses).")
    else:
        cob = meses_cob.groupby("fundo")["mes"].agg(["min", "max", "nunique"])
        st.markdown(f"<span class='nota'>Janela com carteira aberta: {mes_label(meses_cob['mes'].min())} a "
                    f"{mes_label(meses_cob['mes'].max())}. Meses mais recentes ainda estão em sigilo na CVM.</span>",
                    unsafe_allow_html=True)

        st.markdown("#### Quem carregou e quem pesou — top 3 de cada fundo")
        linhas = []
        for f in b.metricas(ini, fim).index:
            x = ca[ca["fundo"] == f]
            if x.empty:
                continue
            top, bot = x.head(3), x.sort_values("contrib").head(3)
            cc = b.concil[(b.concil["fundo"] == f) & (b.concil["mes"] >= ini) & (b.concil["mes"] <= fim)]
            linhas.append({
                "Fundo": f,
                "Ações (soma)": cc["explicado"].sum(),
                "Cota no período": cc["retorno_cota"].sum(),
                "Maiores contribuições": " · ".join(f"{r.ativo} {r.contrib*100:+.1f}" for r in top.itertuples()),
                "Maiores detratores": " · ".join(f"{r.ativo} {r.contrib*100:+.1f}" for r in bot.itertuples()),
                "Meses": int(cob.at[f, "nunique"]) if f in cob.index else 0,
            })
        tabela(pd.DataFrame(linhas), {"Ações (soma)": "{:+.1%}", "Cota no período": "{:+.1%}"}, altura=565)
        st.markdown("<span class='nota'>Valores em p.p. · 'Cota no período' = soma dos retornos mensais dos "
                    "mesmos meses (para comparar com a soma das contribuições).</span>", unsafe_allow_html=True)

        st.markdown("#### Detalhe por fundo")
        fsel = st.selectbox("Fundo", FUNDOS, index=FUNDOS.index(AWR), key="f_atrib")
        x = ca[ca["fundo"] == fsel]
        c1, c2 = st.columns(2)
        for col, dados, titulo in [(c1, x.head(15).iloc[::-1], "15 maiores contribuições"),
                                   (c2, x.sort_values("contrib").head(15).iloc[::-1], "15 maiores detratores")]:
            fig = go.Figure(go.Bar(
                x=dados["contrib"], y=dados["ativo"], orientation="h",
                marker=dict(color=[POS if v >= 0 else NEG for v in dados["contrib"]], cornerradius=4),
                text=[pct(v, 2, True) for v in dados["contrib"]], textposition="outside",
                textfont=dict(color=INK2, size=11),
                customdata=np.c_[dados["peso_medio"], dados["meses"], dados["setor"]],
                hovertemplate="<b>%{y}</b> · %{customdata[2]}<br>Contribuição %{x:+.2%}<br>"
                              "Peso médio %{customdata[0]:.1%} · %{customdata[1]} meses<extra></extra>",
                cliponaxis=False))
            lo, hi = min(dados["contrib"].min(), 0), max(dados["contrib"].max(), 0)
            pad = (hi - lo) * 0.28
            layout(fig, 460, legenda=False, title=dict(text=titulo, font=dict(color=INK, size=14)),
                   xaxis=dict(tickformat=".1%", range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)]))
            col.plotly_chart(fig, width="stretch")

        st.markdown("#### Conciliação mensal: quanto das ações explica a cota")
        cc = b.concil[(b.concil["fundo"] == fsel) & (b.concil["mes"] >= ini) & (b.concil["mes"] <= fim)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cc["mes"], y=cc["explicado"], name="Explicado pelas ações", marker_color=BLUE,
                             marker_cornerradius=4, hovertemplate="%{x|%b/%y}: %{y:+.2%}<extra>ações</extra>"))
        fig.add_trace(go.Bar(x=cc["mes"], y=cc["residuo"], name="Resíduo (hedge, derivativos, giro, custos)",
                             marker_color=GRAY_L, marker_cornerradius=4,
                             hovertemplate="%{x|%b/%y}: %{y:+.2%}<extra>resíduo</extra>"))
        fig.add_trace(go.Scatter(x=cc["mes"], y=cc["retorno_cota"], name="Retorno da cota", mode="lines+markers",
                                 line=dict(color=GOLD if fsel == AWR else INK, width=2), marker=dict(size=8),
                                 hovertemplate="%{x|%b/%y}: %{y:+.2%}<extra>cota</extra>"))
        layout(fig, 380, barmode="relative", bargap=0.35, yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Mapa de contribuição por ação e mês")
        a = b.atrib[(b.atrib["fundo"] == fsel) & (b.atrib["mes"] >= ini) & (b.atrib["mes"] <= fim)]
        top_at = x.reindex(x["contrib"].abs().sort_values(ascending=False).index).head(20)["ativo"]
        hm = a[a["ativo"].isin(top_at)].pivot_table(index="ativo", columns="mes", values="contrib", aggfunc="sum")
        hm = hm.reindex(top_at)
        lim = np.nanmax(np.abs(hm.values)) if hm.size else 0.01
        fig = go.Figure(go.Heatmap(
            z=hm.values, x=[mes_label(d) for d in hm.columns], y=hm.index, colorscale=DIVERG, zmid=0,
            zmin=-lim, zmax=lim, xgap=2, ygap=2,
            text=[[pct(v, 2, True) if pd.notna(v) else "" for v in r] for r in hm.values],
            hovertemplate="%{y} · %{x}: %{text}<extra></extra>", colorbar=dict(tickformat=".1%")))
        layout(fig, 90 + 24 * len(hm), legenda=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Contribuição por setor — todos os fundos")
        cs = b.contrib_setor(ini, fim)
        cs = cs.loc[[f for f in b.metricas(ini, fim).index if f in cs.index]]
        cs = cs[cs.abs().sum().sort_values(ascending=False).index]
        lim = np.nanmax(np.abs(cs.values))
        fig = go.Figure(go.Heatmap(
            z=cs.values, x=cs.columns, y=cs.index, colorscale=DIVERG, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
            text=[[pct(v, 1, True) for v in r] for r in cs.values], texttemplate="%{text}",
            textfont=dict(size=10, color=INK), hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
            colorbar=dict(tickformat=".0%")))
        layout(fig, 90 + 28 * len(cs), legenda=False, yaxis=dict(autorange="reversed"))
        fig.update_yaxes(tickfont=dict(color=INK2))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### As ações que mais fizeram diferença no grupo")
        g = ca.groupby("ativo").agg(contrib_total=("contrib", "sum"), fundos=("fundo", "nunique"),
                                    melhor=("contrib", "max"), setor=("setor", "first")).reset_index()
        awr_c = ca[ca["fundo"] == AWR].set_index("ativo")["contrib"]
        g["AWR"] = g["ativo"].map(awr_c).astype(float)
        g["contrib_media"] = g["contrib_total"] / len(FUNDOS)
        c1, c2 = st.columns(2)
        for col, d, tit in [(c1, g.sort_values("contrib_media", ascending=False).head(12), "Mais ganharam (média do grupo)"),
                            (c2, g.sort_values("contrib_media").head(12), "Mais perderam (média do grupo)")]:
            d = d.rename(columns={"ativo": "Ação", "setor": "Setor", "fundos": "Nº fundos",
                                  "contrib_media": "Média grupo", "AWR": "AWR"})
            col.markdown(f"**{tit}**")
            col.dataframe(d[["Ação", "Setor", "Nº fundos", "Média grupo", "AWR"]].style.format(
                {"Média grupo": "{:+.2%}", "AWR": "{:+.2%}"}, na_rep="–"), hide_index=True, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# 3) CARTEIRAS
# ═════════════════════════════════════════════════════════════════════════════
with abas[2]:
    c1, c2 = st.columns([1.2, 1])
    fsel = c1.selectbox("Fundo", FUNDOS, index=FUNDOS.index(AWR), key="f_cart")
    meses_f = b.meses_abertos.get(fsel, [])
    if not meses_f:
        st.warning("Carteira desse fundo não está aberta na CVM no período.")
    else:
        dsel = c2.selectbox("Mês da carteira", meses_f[::-1], format_func=mes_label, key="m_cart")
        cart = b.carteira(fsel, dsel)
        res = b.resumo_ok[(b.resumo_ok["fundo"] == fsel) & (b.resumo_ok["data"] == dsel)].iloc[0]
        k = st.columns(6)
        kpi(k[0], "PL", f"R$ {num(res['pl']/1e6, 0)} mi")
        kpi(k[1], "Ações compradas", pct(res["pct_acoes"], 0))
        kpi(k[2], "Vendido (aluguel)", pct(res["pct_short"], 0))
        kpi(k[3], "Offshore (não aberto)", pct(res["pct_offshore"], 0))
        kpi(k[4], "Caixa / RF", pct(res["pct_caixa"], 0))
        kpi(k[5], "Nº de ações (>0,5%)", str(int((cart["peso"] > 0.005).sum())))
        st.markdown("")

        c1, c2 = st.columns([1.4, 1])
        nxt = dsel + pd.offsets.MonthEnd(1)
        cart = cart.assign(ret_mes_seg=cart["ativo"].map(b.ret_ativo.loc[nxt]) if nxt in b.ret_ativo.index else np.nan)
        cart = cart.assign(contrib_seg=cart["peso"] * cart["ret_mes_seg"])
        with c1:
            st.markdown(f"**Posições em {mes_label(dsel)}** (look-through)")
            v = cart[cart["peso"].abs() > 0.0005][["ativo", "desc", "setor", "peso", "valor", "ret_mes_seg", "contrib_seg"]]
            v = v.rename(columns={"ativo": "Ativo", "desc": "Descrição", "setor": "Setor", "peso": "% PL",
                                  "valor": "R$ mi", "ret_mes_seg": f"Retorno {mes_label(nxt)}",
                                  "contrib_seg": "Contrib."})
            v["R$ mi"] = v["R$ mi"] / 1e6
            st.dataframe(v.style.format({"% PL": "{:.2%}", "R$ mi": "{:,.1f}", f"Retorno {mes_label(nxt)}": "{:+.1%}",
                                         "Contrib.": "{:+.2%}"}, na_rep="–"),
                         hide_index=True, width="stretch", height=520)
        with c2:
            st.markdown("**Alocação por setor (% PL, líquido de vendidos)**")
            s = cart.groupby("setor")["peso"].sum().sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                   marker=dict(color=[GOLD if fsel == AWR else BLUE if v >= 0 else NEG for v in s.values],
                                               cornerradius=4),
                                   text=[pct(v, 1) for v in s.values], textposition="outside",
                                   textfont=dict(color=INK2),
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            layout(fig, 520, legenda=False, xaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Evolução da exposição")
        r = b.resumo_ok[(b.resumo_ok["fundo"] == fsel) & b.resumo_ok["aberto"]].sort_values("data")
        fig = go.Figure()
        for colname, nome, cor in [("pct_acoes", "Ações compradas", GOLD if fsel == AWR else BLUE),
                                   ("pct_offshore", "Offshore", AQUA), ("pct_caixa", "Caixa/RF", GRAY_L),
                                   ("pct_short", "Vendido (aluguel)", NEG)]:
            fig.add_trace(go.Bar(x=r["data"], y=r[colname], name=nome, marker_color=cor, marker_cornerradius=4,
                                 hovertemplate=f"{nome} %{{x|%b/%y}}: %{{y:.1%}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=r["data"], y=r["pct_acoes"] + r["pct_short"], name="Líquido em ações (ex-futuros)",
                                 mode="lines+markers", line=dict(color=INK, width=2), marker=dict(size=8),
                                 hovertemplate="Líquido %{x|%b/%y}: %{y:.1%}<extra></extra>"))
        layout(fig, 380, barmode="relative", bargap=0.35, yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Top 10 ao longo do tempo (% PL)")
        e = b.eq[(b.eq["fundo"] == fsel) & (b.eq["data"].isin(meses_f))]
        top = e.groupby("ativo")["peso"].mean().sort_values(ascending=False).head(15).index
        hm = e[e["ativo"].isin(top)].pivot_table(index="ativo", columns="data", values="peso", aggfunc="sum").reindex(top)
        fig = go.Figure(go.Heatmap(
            z=hm.values, x=[mes_label(d) for d in hm.columns], y=hm.index,
            colorscale=[[0, "#1d2a3a"], [1, "#3987e5"]], xgap=2, ygap=2,
            text=[[pct(v, 1) if pd.notna(v) else "" for v in r] for r in hm.values], texttemplate="%{text}",
            textfont=dict(size=10, color=INK), hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
            colorbar=dict(tickformat=".0%")))
        layout(fig, 90 + 26 * len(hm), legenda=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Concentração e número de posições — todos os fundos (último mês aberto)")
    cc = b.concentracao()
    ult = cc.sort_values("data").groupby("fundo").tail(1)
    ult = ult.rename(columns={"fundo": "Fundo", "data": "Mês", "top5": "Top 5", "top10": "Top 10",
                              "n": "Nº ações >0,5%", "n_efetivo": "Nº efetivo (1/HHI)", "pct_acoes": "Long",
                              "pct_short": "Short", "pct_offshore": "Offshore", "pct_caixa": "Caixa"})
    ult["Líquido (ex-futuros)"] = ult["Long"] + ult["Short"]
    ult["Mês"] = ult["Mês"].map(mes_label)
    tabela(ult[["Fundo", "Mês", "Long", "Short", "Líquido (ex-futuros)", "Offshore", "Caixa", "Top 5", "Top 10", "Nº ações >0,5%",
                "Nº efetivo (1/HHI)"]].sort_values("Top 10", ascending=False),
           {"Long": "{:.0%}", "Short": "{:.0%}", "Líquido (ex-futuros)": "{:.0%}", "Offshore": "{:.0%}", "Caixa": "{:.0%}",
            "Top 5": "{:.0%}", "Top 10": "{:.0%}", "Nº efetivo (1/HHI)": "{:.1f}"}, altura=565)

# ═════════════════════════════════════════════════════════════════════════════
# 4) CONSENSO
# ═════════════════════════════════════════════════════════════════════════════
with abas[4]:
    todos_meses = sorted({m for v in b.meses_abertos.values() for m in v})
    ult_peers = sorted(m[-1] for f, m in b.meses_abertos.items() if f != AWR and m)
    padrao = ult_peers[len(ult_peers) // 2] if ult_peers else todos_meses[-1]
    opcoes = todos_meses[::-1]
    dref = st.selectbox("Carteiras de referência (último mês aberto de cada fundo até…)", opcoes,
                        index=opcoes.index(padrao) if padrao in opcoes else 0, format_func=mes_label, key="m_cons")
    cons, tot, n_f = b.consenso(dref)
    st.markdown(f"<span class='nota'>{n_f} fundos com carteira aberta até {mes_label(dref)}.</span>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Ações mais compartilhadas (crowding)")
        top = cons.head(20).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top["n_long"], y=top["ativo"], orientation="h", name="Nº de fundos comprados",
                             marker=dict(color=BLUE, cornerradius=4),
                             customdata=np.c_[top["peso_medio"], top["peso_awr"], top["setor"]],
                             text=[f"{n} · peso médio {pct(p,1)}" + (f" · <b>AWR {pct(w,1)}</b>" if w > 0.002 else "")
                                   for n, p, w in zip(top["n_long"], top["peso_medio"], top["peso_awr"])],
                             textposition="outside", textfont=dict(color=INK2, size=11),
                             hovertemplate="<b>%{y}</b> · %{customdata[2]}<br>%{x} fundos comprados<br>"
                                           "peso médio (quem tem) %{customdata[0]:.1%}<br>AWR %{customdata[1]:.1%}<extra></extra>"))
        layout(fig, 620, legenda=False, xaxis=dict(title="nº de fundos com posição comprada", range=[0, n_f + 6]))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### AWR vs média dos peers (p.p. do PL)")
        d = cons[(cons["peso_awr"].abs() > 0.002) | (cons["n_long"] >= 5)].copy()
        d = pd.concat([d.nlargest(10, "ativo_vs_peers"), d.nsmallest(10, "ativo_vs_peers")]).drop_duplicates("ativo")
        d = d.sort_values("ativo_vs_peers")
        fig = go.Figure(go.Bar(
            x=d["ativo_vs_peers"], y=d["ativo"], orientation="h",
            marker=dict(color=[GOLD if v > 0 else GRAY_L for v in d["ativo_vs_peers"]], cornerradius=4),
            text=[pct(v, 1, True).replace("%", " p.p.") for v in d["ativo_vs_peers"]], textposition="outside",
            textfont=dict(color=INK2, size=11), customdata=np.c_[d["peso_awr"], d["n_long"]],
            hovertemplate="<b>%{y}</b><br>AWR %{customdata[0]:.1%} · %{customdata[1]} peers comprados<br>"
                          "diferença %{x:+.1%}<extra></extra>", cliponaxis=False))
        lo, hi = d["ativo_vs_peers"].min(), d["ativo_vs_peers"].max()
        pad = (hi - lo) * 0.18
        layout(fig, 620, legenda=False, xaxis=dict(tickformat=".0%", range=[min(lo, 0) - pad, max(hi, 0) + pad]))
        st.plotly_chart(fig, width="stretch")
        st.markdown("<span class='nota'>Dourado = AWR acima da média dos peers (overweight); "
                    "cinza = abaixo (underweight/sem posição).</span>", unsafe_allow_html=True)

    st.markdown("#### Semelhança entre carteiras (cosseno dos pesos comprados)")
    sim = b.similaridade(tot)
    ordem = sim[AWR].sort_values(ascending=False).index if AWR in sim else sim.index
    sim = sim.loc[ordem, ordem]
    fig = go.Figure(go.Heatmap(
        z=sim.values, x=sim.columns, y=sim.index, colorscale=[[0, "#1d2a3a"], [1, "#3987e5"]], zmin=0, zmax=1,
        xgap=2, ygap=2, text=[[f"{v:.0%}" for v in r] for r in sim.values], texttemplate="%{text}",
        textfont=dict(size=10, color=INK), hovertemplate="%{y} × %{x}: %{text}<extra></extra>"))
    layout(fig, 560, legenda=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Exclusivas do AWR (nenhum ou 1 peer comprado)")
        ex = cons[(cons["peso_awr"] > 0.003) & (cons["n_long"] <= 2)].sort_values("peso_awr", ascending=False)
        st.dataframe(ex[["ativo", "setor", "peso_awr", "n_long"]].rename(columns={
            "ativo": "Ação", "setor": "Setor", "peso_awr": "Peso AWR", "n_long": "Fundos c/ posição"}).style.format(
            {"Peso AWR": "{:.1%}"}), hide_index=True, width="stretch")
    with c2:
        st.markdown("#### Consenso dos peers que o AWR não tem")
        fa = cons[(cons["peso_awr"] <= 0.001) & (cons["n_long"] >= 4)].sort_values(["n_long", "peso_medio"], ascending=False)
        st.dataframe(fa[["ativo", "setor", "n_long", "peso_medio"]].head(15).rename(columns={
            "ativo": "Ação", "setor": "Setor", "n_long": "Nº peers", "peso_medio": "Peso médio"}).style.format(
            {"Peso médio": "{:.1%}"}), hide_index=True, width="stretch")

    st.markdown("#### Matriz de pesos (% PL)")
    mt = tot.reindex(cons.head(30)["ativo"])
    mt = mt[[f for f in FUNDOS if f in mt.columns]]
    lim = 0.12
    fig = go.Figure(go.Heatmap(
        z=mt.values, x=mt.columns, y=mt.index, colorscale=DIVERG, zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
        text=[[pct(v, 1) if abs(v) > 0.0005 else "" for v in r] for r in mt.values], texttemplate="%{text}",
        textfont=dict(size=9, color=INK), hovertemplate="%{x} · %{y}: %{text}<extra></extra>",
        colorbar=dict(tickformat=".0%")))
    layout(fig, 120 + 22 * len(mt), legenda=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# 5) MOVIMENTAÇÕES
# ═════════════════════════════════════════════════════════════════════════════
with abas[5]:
    st.markdown("Variação de peso **descontado o efeito preço**: compara o peso do mês com o que o ativo "
                "teria se o gestor não tivesse operado.")
    fsel = st.selectbox("Fundo", FUNDOS, index=FUNDOS.index(AWR), key="f_mov")
    mv = b.movimentos(fsel)
    if mv.empty:
        st.info("Sem meses consecutivos com carteira aberta para esse fundo.")
    else:
        mv = mv[(mv["mes"] >= ini) & (mv["mes"] <= fim)]
        ms = sorted(mv["mes"].unique())[::-1]
        msel = st.selectbox("Mês", ms, format_func=mes_label, key="m_mov")
        x = mv[mv["mes"] == msel]
        cols = st.columns(4)
        for col, tipo in zip(cols, ["Nova posição", "Aumentou", "Reduziu", "Zerou"]):
            y = x[x["acao"] == tipo].sort_values("delta_ativo", ascending=tipo in ("Reduziu", "Zerou"))
            col.markdown(f"**{tipo}** ({len(y)})")
            col.dataframe(y[["ativo", "peso_antes", "peso_depois"]].rename(columns={
                "ativo": "Ação", "peso_antes": "Antes", "peso_depois": "Depois"}).style.format(
                {"Antes": "{:.1%}", "Depois": "{:.1%}"}), hide_index=True, width="stretch", height=330)

    st.markdown("#### Fluxo do grupo: o que os peers mais compraram e venderam")
    todos_mv = []
    for f in FUNDOS:
        m = b.movimentos(f)
        if not m.empty:
            todos_mv.append(m.assign(fundo=f))
    if todos_mv:
        tm = pd.concat(todos_mv)
        tm = tm[(tm["mes"] >= ini) & (tm["mes"] <= fim)]
        ult_peers = sorted(m[-1] for f, m in b.meses_abertos.items() if f != AWR and m)
        op_g = sorted(tm["mes"].unique())[::-1]
        pad_g = ult_peers[len(ult_peers) // 2] if ult_peers else None
        mes_g = st.selectbox("Mês", op_g, index=op_g.index(pad_g) if pad_g in op_g else 0,
                             format_func=mes_label, key="m_mov_g")
        t = tm[tm["mes"] == mes_g]
        agg = t.groupby("ativo").agg(delta=("delta_ativo", "sum"),
                                     compradores=("delta_ativo", lambda s: int((s > 0.003).sum())),
                                     vendedores=("delta_ativo", lambda s: int((s < -0.003).sum())))
        agg["saldo"] = agg["compradores"] - agg["vendedores"]
        d = pd.concat([agg.nlargest(12, "saldo"), agg.nsmallest(12, "saldo")]).drop_duplicates()
        d = d.sort_values(["saldo", "delta"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["compradores"], y=d.index, orientation="h", name="Fundos aumentando",
                             marker=dict(color=POS, cornerradius=4), hovertemplate="%{y}: %{x} aumentaram<extra></extra>"))
        fig.add_trace(go.Bar(x=-d["vendedores"], y=d.index, orientation="h", name="Fundos reduzindo",
                             marker=dict(color=NEG, cornerradius=4),
                             customdata=d["vendedores"], hovertemplate="%{y}: %{customdata} reduziram<extra></extra>"))
        layout(fig, 640, barmode="relative", xaxis=dict(title="nº de fundos (← reduzindo | aumentando →)", dtick=1))
        st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# 6) CAPTAÇÃO & PL
# ═════════════════════════════════════════════════════════════════════════════
with abas[6]:
    fl = b.fluxos_mensais()
    fl = fl[(fl["mes"] >= ini.to_period("M").to_timestamp("M")) & (fl["mes"] <= fim)]
    tot_f = fl.groupby("fundo")["liquida"].sum().sort_values()
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("#### Captação líquida no período (R$ mi)")
        fig = go.Figure(go.Bar(
            x=tot_f.values / 1e6, y=tot_f.index, orientation="h",
            marker=dict(color=[GOLD if f == AWR else (POS if v >= 0 else NEG) for f, v in tot_f.items()], cornerradius=4),
            text=[num(v / 1e6, 0) for v in tot_f.values], textposition="outside", textfont=dict(color=INK2),
            hovertemplate="%{y}: R$ %{x:,.1f} mi<extra></extra>", cliponaxis=False))
        lo, hi = min(tot_f.min() / 1e6, 0), max(tot_f.max() / 1e6, 0)
        layout(fig, 520, legenda=False, xaxis=dict(range=[lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.12]))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### Evolução do PL (base 100)")
        pl = fl.pivot_table(index="mes", columns="fundo", values="pl")
        pl = pl / pl.bfill().iloc[0] * 100
        fig = go.Figure()
        for f in [x for x in FUNDOS if x != AWR and x not in destaques] + destaques + [AWR]:
            if f not in pl:
                continue
            forte = f == AWR or f in destaques
            fig.add_trace(go.Scatter(x=pl.index, y=pl[f], name=f, mode="lines",
                                     line=dict(color=cor_fundo(f, destaques), width=2.4 if forte else 1.1),
                                     opacity=1 if forte else 0.55, showlegend=forte,
                                     hovertemplate=f"{f} %{{x|%b/%y}}: %{{y:.0f}}<extra></extra>"))
        layout(fig, 520, yaxis=dict(type="log", title="PL base 100 (escala log)", tickvals=[25, 50, 100, 200, 400, 800, 1600], ticktext=["25", "50", "100", "200", "400", "800", "1.600"]))
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Captação líquida mensal por fundo (R$ mi)")
    hm = fl.pivot_table(index="fundo", columns="mes", values="liquida") / 1e6
    hm = hm.loc[tot_f.index[::-1]]
    lim = np.nanpercentile(np.abs(hm.values), 95)
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=[mes_label(d) for d in hm.columns], y=hm.index, colorscale=DIVERG, zmid=0, zmin=-lim, zmax=lim,
        xgap=2, ygap=2, text=[[num(v, 0) for v in r] for r in hm.values], texttemplate="%{text}",
        textfont=dict(size=10, color=INK), hovertemplate="%{y} · %{x}: R$ %{text} mi<extra></extra>"))
    layout(fig, 90 + 28 * len(hm), legenda=False, yaxis=dict(autorange="reversed"))
    fig.update_yaxes(tickfont=dict(color=INK2))
    st.plotly_chart(fig, width="stretch")

    ult = fl.sort_values("mes").groupby("fundo").tail(1).set_index("fundo")
    pri = fl.sort_values("mes").groupby("fundo").head(1).set_index("fundo")
    t = pd.DataFrame({"PL atual (R$ mi)": ult["pl"] / 1e6,
                      "Captação (R$ mi)": fl.groupby("fundo")["captacao"].sum() / 1e6,
                      "Resgates (R$ mi)": fl.groupby("fundo")["resgate"].sum() / 1e6,
                      "Líquida (R$ mi)": tot_f / 1e6,
                      "Cotistas": ult["cotistas"], "Var. cotistas": ult["cotistas"] - pri["cotistas"]})
    t = t.rename_axis("Fundo").reset_index()
    tabela(t.sort_values("PL atual (R$ mi)", ascending=False),
           {"PL atual (R$ mi)": "{:,.0f}", "Captação (R$ mi)": "{:,.0f}", "Resgates (R$ mi)": "{:,.0f}",
            "Líquida (R$ mi)": "{:+,.0f}", "Cotistas": "{:,.0f}", "Var. cotistas": "{:+,.0f}"}, altura=565)

# ═════════════════════════════════════════════════════════════════════════════
# 7) METODOLOGIA
# ═════════════════════════════════════════════════════════════════════════════
with abas[7]:
    conf = b.resumo.pivot_table(index="fundo", columns="data", values="pct_confid")
    st.markdown(f"""
**Fontes (todas públicas):**
- **Cotas, PL, captação, resgate e cotistas:** Informe Diário da CVM (`inf_diario_fi`). Fundos com mais de uma
  subclasse usam, a cada dia, a subclasse de maior PL (série emendada por retornos); PL e fluxos somam as subclasses.
- **Carteiras:** CDA mensal da CVM (`cda_fi`). Fundos que investem via FIC → master são **abertos até o ativo
  final** (look-through), proporcionalmente à participação no PL do master. Quando o próprio vínculo FIC → master
  está em sigilo, usa-se o vínculo observado em outro mês.
- **Preços:** {b.fonte_preco}. Desdobramentos e grupamentos são detectados no preço implícito.
- **CDI:** BCB/SGS série 12 · **Ibovespa:** Yahoo Finance.

**Sigilo:** a CVM permite que o gestor omita a carteira por até 6 meses (bloco CONFID). Meses com mais de
{int(25)}% do PL em sigilo ficam fora da atribuição. O painel é atualizado conforme o sigilo vence.
{"O AWR é exibido com o mesmo corte temporal dos peers." if MESMO_CORTE_AWR else ""}

**Atribuição:** contribuição = peso da ação no fim de *m-1* × retorno da ação em *m* (inclui vendidos
com peso negativo). Não captura operações dentro do mês, derivativos (futuros/opções), offshore não aberto,
custos e taxas. Essa diferença aparece como **resíduo** na conciliação. A soma das contribuições é aritmética
(aproximação).

**Movimentações:** peso atual × peso que o ativo teria sem negociação (peso anterior corrigido pela
variação de preço). Diferença acima de 0,5 p.p. = aumento/redução.

**Classificação dos ativos (bloco 4 do CDA):** "Obrigações por ações recebidas em empréstimo" = vendido (short);
"cedidas em empréstimo" = comprado (alugado para terceiros). Opções, futuros e termo aparecem nas exposições,
mas não entram no peso em ações.

**Aviso:** material informativo, baseado exclusivamente em dados públicos. Não é recomendação de investimento.
Rentabilidade passada não garante rentabilidade futura.
""")
    st.markdown("#### Parcela da carteira em sigilo por mês")
    fig = go.Figure(go.Heatmap(
        z=conf.values, x=[mes_label(d) for d in conf.columns], y=conf.index,
        colorscale=[[0, "#1d2a3a"], [1, "#3987e5"]], zmin=0, zmax=1, xgap=2, ygap=2,
        text=[[pct(min(v, 1), 0) if pd.notna(v) else "" for v in r] for r in conf.values], texttemplate="%{text}",
        textfont=dict(size=9, color=INK), hovertemplate="%{y} · %{x}: %{text} em sigilo<extra></extra>"))
    layout(fig, 90 + 26 * len(conf), legenda=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# 4) PROJEÇÃO (HOJE)
# ═════════════════════════════════════════════════════════════════════════════
with abas[3]:
    st.markdown(
        "Estimativa da carteira **atual** a partir da última carteira aberta na CVM e da **cota diária** "
        "divulgada depois dela. Hipótese central: o gestor mexeu pouco. A cota diz o quanto isso é verdade.")
    rb = pj.resumo_backtest()
    ep = pj.erro_por_fundo()
    nome_modelo = {"escala": "escala do book + hedge Ibovespa", "setor": "multiplicador por setor + hedge",
                   "ativo": "peso por ativo + hedge"}[pj.modelo]
    if not rb.empty:
        k = st.columns(4)
        r3 = rb.set_index("meses")
        m3 = r3.loc[3] if 3 in r3.index else r3.iloc[0]
        kpi(k[0], "Modelo escolhido no backtest", nome_modelo.split(" +")[0].capitalize(), nome_modelo)
        kpi(k[1], "Erro médio de peso por ação", f"{m3['mae_cal']*100:.1f}".replace(".", ",") + " p.p.",
            f"projeção de 3 meses · {int(m3['janelas'])} janelas testadas")
        kpi(k[2], "Erro na exposição total", f"{m3['erro_exp_cal']*100:.0f} p.p.",
            f"carteira congelada: {m3['erro_exp_congelada']*100:.0f} p.p.")
        kpi(k[3], "PL em papéis novos após 3 meses", pct(m3["novos"], 0), "o que nenhuma projeção enxerga")
    st.markdown("")

    st.markdown("#### Aderência: a cota real ainda se parece com a última carteira aberta?")
    tf = pj.tabela_fundos()
    if not tf.empty:
        tf = tf.merge(ep[["erro_exp"]].rename(columns={"erro_exp": "Erro hist. exposição"}),
                      left_on="Fundo", right_index=True, how="left")
        tf["Base"] = tf["Base"].map(mes_label)
        tf = tf.sort_values("TE congelada").rename(columns={"Hedge Ibov est.": "Ibov adicional est."})
        tabela(tf[["Fundo", "Base", "Pregões", "Cota real", "Carteira congelada", "Diferença", "TE congelada",
                   "Correlação", "Exposição base", "Exposição estimada", "Ibov adicional est.",
                   "Erro hist. exposição", "Aderência"]],
               {"Cota real": "{:+.1%}", "Carteira congelada": "{:+.1%}", "Diferença": "{:+.1%}",
                "TE congelada": "{:.1%}", "Correlação": "{:.2f}", "Exposição base": "{:.0%}",
                "Exposição estimada": "{:.0%}", "Ibov adicional est.": "{:+.0%}", "Erro hist. exposição": "{:.0%}"},
               altura=565)
        st.markdown("<span class='nota'>TE congelada = tracking error anualizado entre a cota real e a carteira-base "
                    "mantida parada. Quanto menor, mais a carteira atual deve se parecer com a base. "
                    "'Ibov adicional' positivo = exposição extra fora da carteira (futuro comprado, offshore, papéis "
                    "novos); negativo = hedge.</span>", unsafe_allow_html=True)

    st.markdown("#### Detalhe por fundo")
    fundos_pj = [f for f in FUNDOS if pj.resultados.get(f)]
    fsel = st.selectbox("Fundo", fundos_pj, index=fundos_pj.index(AWR) if AWR in fundos_pj else 0, key="f_proj")
    r = pj.resultados.get(fsel)
    if r:
        k = st.columns(5)
        kpi(k[0], "Carteira-base", mes_label(r["t0"]), f"{r['dias']} pregões até {r['t1']:%d/%m/%y}")
        kpi(k[1], "Cota real desde a base", pct(r["ret_real"], 1, True))
        kpi(k[2], "Carteira congelada", pct(r["ret_congelada"], 1, True),
            f"diferença {pct(r['ret_real'] - r['ret_congelada'], 1, True).replace('%', ' p.p.')}")
        kpi(k[3], "Exposição em ações", f"{pct(r['exp_base'], 0)} → {pct(r['exp_estimada'], 0)}",
            "base → estimada hoje")
        kpi(k[4], "Ibov adicional estimado", pct(r["hedge"], 0, True),
            "futuro/offshore/papéis novos" if r["hedge"] >= 0 else "hedge vendido")

        fig = go.Figure()
        cor = GOLD if fsel == AWR else BLUE
        fig.add_trace(go.Scatter(x=r["real"].index, y=r["real"] - 1, name="Cota real", line=dict(color=cor, width=2.4),
                                 hovertemplate="Cota real %{x|%d/%m/%y}: %{y:+.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=r["congelada"].index, y=r["congelada"] - 1,
                                 name=f"Carteira de {mes_label(r['t0'])} congelada",
                                 line=dict(color=INK2, width=1.8, dash="dash"),
                                 hovertemplate="Congelada %{x|%d/%m/%y}: %{y:+.1%}<extra></extra>"))
        if b.ibov_idx is not None:
            ib = b.ibov_idx[b.ibov_idx.index >= r["real"].index[0]].dropna()
            ib = ib / ib.iloc[0]
            fig.add_trace(go.Scatter(x=ib.index, y=ib - 1, name="Ibovespa", line=dict(color=INK3, width=1.2, dash="dot"),
                                     hovertemplate="Ibovespa %{x|%d/%m/%y}: %{y:+.1%}<extra></extra>"))
        layout(fig, 400, hovermode="x unified", yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")

        pw = r["pesos"].copy()
        c1, c2 = st.columns([1.35, 1])
        with c1:
            st.markdown(f"**Carteira estimada hoje** (a partir de {mes_label(r['t0'])})")
            v = pw[["setor", "peso_base", "peso_congelado", "peso_estimado", "ret_desde_base", "contrib_desde_base"]]
            v = v[v[["peso_base", "peso_estimado"]].abs().max(axis=1) > 0.003]
            v = v.rename_axis("Ativo").reset_index().rename(columns={
                "setor": "Setor", "peso_base": f"Peso {mes_label(r['t0'])}", "peso_congelado": "Só efeito preço",
                "peso_estimado": "Estimado hoje", "ret_desde_base": "Retorno desde a base",
                "contrib_desde_base": "Contrib. desde a base"})
            fmt = {c: "{:.1%}" for c in v.columns if c.startswith(("Peso", "Só", "Estimado"))}
            fmt.update({"Retorno desde a base": "{:+.1%}", "Contrib. desde a base": "{:+.2%}"})
            st.dataframe(v.style.format(fmt, na_rep="–"), hide_index=True, width="stretch", height=520)
        with c2:
            st.markdown("**O que deve ter carregado desde a base**")
            cb = pw["contrib_desde_base"].dropna().sort_values()
            d = pd.concat([cb.head(8), cb.tail(8)])
            d = d[~d.index.duplicated()].sort_values()
            fig = go.Figure(go.Bar(
                x=d.values, y=d.index, orientation="h",
                marker=dict(color=[POS if x >= 0 else NEG for x in d.values], cornerradius=4),
                text=[pct(x, 2, True) for x in d.values], textposition="outside", textfont=dict(color=INK2, size=11),
                cliponaxis=False, hovertemplate="%{y}: %{x:+.2%} do PL<extra></extra>"))
            lo, hi = min(d.min(), 0), max(d.max(), 0)
            layout(fig, 520, legenda=False,
                   xaxis=dict(tickformat=".1%", range=[lo - (hi - lo) * .25, hi + (hi - lo) * .25]))
            st.plotly_chart(fig, width="stretch")
        st.markdown("<span class='nota'>Contribuição desde a base = peso na carteira aberta × retorno do papel "
                    "até hoje (buy-and-hold). 'Só efeito preço' = peso que o papel teria hoje se o gestor não "
                    "tivesse operado; 'Estimado hoje' aplica a escala/hedge que a cota diária indica.</span>",
                    unsafe_allow_html=True)

    st.markdown("#### Quem deve estar carregando cada fundo desde o sigilo")
    linhas = []
    for f in FUNDOS:
        rr = pj.resultados.get(f)
        if not rr:
            continue
        cb = rr["pesos"]["contrib_desde_base"].dropna().sort_values(ascending=False)
        linhas.append({"Fundo": f, "Base": mes_label(rr["t0"]), "Cota real": rr["ret_real"],
                       "Carteira congelada": rr["ret_congelada"],
                       "Maiores contribuições": " · ".join(f"{a} {v*100:+.1f}" for a, v in cb.head(3).items()),
                       "Maiores detratores": " · ".join(f"{a} {v*100:+.1f}" for a, v in cb.tail(3)[::-1].items())})
    tabela(pd.DataFrame(linhas).sort_values("Cota real", ascending=False),
           {"Cota real": "{:+.1%}", "Carteira congelada": "{:+.1%}"}, altura=565)

    st.markdown("#### Consenso projetado para hoje")
    cp, totp = pj.consenso_projetado()
    cp = cp[cp["ativo"].str.fullmatch(r"[A-Z]{4}\d{1,2}")].head(25)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        d = cp.iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["peso_medio_base"], y=d["ativo"], orientation="h", name="Peso médio dos peers na base",
                             marker=dict(color=GRAY, cornerradius=4),
                             hovertemplate="%{y}: %{x:.2%} na base<extra></extra>"))
        fig.add_trace(go.Bar(x=d["peso_medio_peers"], y=d["ativo"], orientation="h", name="Peso médio estimado hoje",
                             marker=dict(color=BLUE, cornerradius=4), customdata=np.c_[d["n_fundos"], d["peso_awr"]],
                             hovertemplate="%{y}: %{x:.2%} hoje · %{customdata[0]} peers · "
                                           "AWR %{customdata[1]:.1%}<extra></extra>"))
        layout(fig, 640, barmode="group", bargap=0.25, bargroupgap=0.1, xaxis=dict(tickformat=".1%"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        t = cp.rename(columns={"ativo": "Ação", "setor": "Setor", "n_fundos": "Nº peers",
                               "peso_medio_peers": "Peso médio hoje", "var_media": "Var. vs base",
                               "peso_awr": "AWR est."})
        t = t.rename(columns={"peso_medio_congelado": "Só efeito preço"})
        st.dataframe(t[["Ação", "Setor", "Nº peers", "Só efeito preço", "Peso médio hoje", "Var. vs base",
                        "AWR est."]].style.format(
            {"Só efeito preço": "{:.1%}", "Peso médio hoje": "{:.1%}", "Var. vs base": "{:+.1%}", "AWR est.": "{:.1%}"}),
            hide_index=True, width="stretch", height=640)

    with st.expander("Como a projeção é feita e quão confiável ela é"):
        novos_txt = pct(rb["novos"].mean(), 0) if not rb.empty else "–"
        st.markdown(f"""
1. **Carteira-base:** última carteira aberta de cada fundo (look-through), com pesos derivando pelo preço
   diário (ajustado por proventos) até hoje. Papéis sem preço diário entram com proxy Ibovespa; caixa rende CDI;
   offshore e derivativos ficam parados.
2. **Calibração pela cota:** nos últimos 63 pregões, o retorno diário da cota (menos CDI) é regredido contra a
   contribuição da carteira-base e o Ibovespa. Modelo escolhido: **{nome_modelo}**, com âncora na carteira
   congelada (ridge, λ = {pj.lam:g}). Ele estima quanto o gestor aumentou ou reduziu o book e o hedge.
3. **Backtest:** o mesmo procedimento aplicado a janelas passadas em que a carteira seguinte é pública
   (3 e 6 meses à frente). Foram testados modelos por ativo, por setor e por escala; o escolhido tem o menor erro.
   Pesos individuais não são identificáveis só pela cota. A projeção por ação é essencialmente a carteira
   congelada, e o ganho real está na **exposição total e no hedge**.
4. **Limites:** papéis comprados depois da base não aparecem (em média ~{novos_txt} do PL após 3–6 meses).
   Fundos com offshore relevante (Truxt, Navi) ou derivativos (Opportunity) têm aderência baixa, e a projeção
   deles deve ser lida com cautela.
""")
        if not rb.empty:
            st.dataframe(rb.rename(columns={"meses": "Horizonte (meses)", "janelas": "Janelas",
                                            "mae_congelada": "Erro peso congelada", "mae_cal": "Erro peso modelo",
                                            "acerto": "Acerto direção", "erro_exp_congelada": "Erro exposição congelada",
                                            "erro_exp_cal": "Erro exposição modelo", "novos": "PL em papéis novos",
                                            "ganho": "Ganho no peso"}).style.format(
                {"Erro peso congelada": "{:.2%}", "Erro peso modelo": "{:.2%}", "Acerto direção": "{:.0%}",
                 "Erro exposição congelada": "{:.1%}", "Erro exposição modelo": "{:.1%}",
                 "PL em papéis novos": "{:.0%}", "Ganho no peso": "{:+.0%}"}), hide_index=True, width="stretch")
