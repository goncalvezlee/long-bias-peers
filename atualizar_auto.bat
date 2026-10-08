@echo off
REM ============================================================
REM  Peers Long Bias - versao para o Agendador de Tarefas
REM  (sem pause; grava log em logs\atualizacao.log)
REM ============================================================
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\atualizacao.log
echo ===== %date% %time% ===== >> %LOG%

echo [1/2] Coletando dados... >> %LOG%
python coletor_cvm.py >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERRO] Coleta falhou. Nada foi publicado. >> %LOG%
    exit /b 1
)

echo [2/2] Enviando para o GitHub... >> %LOG%
git add -A >> %LOG% 2>&1
git commit -m "Atualizacao de dados %date%" -q >> %LOG% 2>&1
git pull --rebase -q origin main >> %LOG% 2>&1
git push -q origin main >> %LOG% 2>&1
if errorlevel 1 (
    echo [ERRO] Push falhou. >> %LOG%
    exit /b 1
)
echo OK >> %LOG%
