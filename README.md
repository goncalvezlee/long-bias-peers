# Long Bias · Raio-X dos Peers

Painel com dados **públicos da CVM** comparando o AWR Long Bias FIA com 14 fundos
long biased de referência, desde 10/04/2025:

- Rentabilidade, risco, drawdown, Sharpe, beta e ranking por período
- **Atribuição de performance**: quais ações carregaram (e quais pesaram) cada fundo
- Carteiras com look-through (FIC → master), exposição long/short/offshore/caixa
- Consenso (crowding), semelhança entre carteiras e AWR vs média dos peers
- Movimentações mês a mês (descontado o efeito preço) e fluxo do grupo
- Captação líquida, PL e cotistas

Fontes: CVM (Informe Diário e CDA), Yahoo Finance e BCB. A CVM permite sigilo de
carteira por até 6 meses; o painel avança conforme o sigilo vence.

Material informativo. Não é recomendação de investimento.

## Atualização (local)

```
python coletor_cvm.py      # baixa CVM/preços e regrava data/
streamlit run app.py
```
