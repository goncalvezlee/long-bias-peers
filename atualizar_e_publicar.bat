@echo off
REM ============================================================
REM  Peers Long Bias - atualiza dados da CVM e publica
REM  (GitHub -> Streamlit Cloud reconstroi o site sozinho)
REM ============================================================
cd /d "%~dp0"
echo [1/2] Coletando dados (CVM, precos, CDI, backtest)...
python coletor_cvm.py
if errorlevel 1 (
    echo [ERRO] Coleta falhou. Nada foi publicado.
    pause
    exit /b 1
)
echo [2/2] Enviando para o GitHub...
git add -A
git commit -m "Atualizacao de dados %date%" -q
git pull --rebase -q origin main
git push origin main
if errorlevel 1 (
    echo [ERRO] Push falhou.
    pause
    exit /b 1
)
echo OK - o Streamlit Cloud atualiza o site em 1-2 min.
pause
