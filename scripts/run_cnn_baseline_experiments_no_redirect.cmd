@echo off
REM Run UNet and DeepLabV3-like baselines without stdout/stderr redirection.
REM This avoids the Windows torch shm.dll WinError 1114 observed when Python output is redirected.

cd /d "%~dp0\.."
set TORCHDYNAMO_DISABLE=1
set NO_ALBUMENTATIONS_UPDATE=1
set PYTHONDONTWRITEBYTECODE=1
set PYTHONPATH=
set PYTHONHOME=

if not exist logs\baselines mkdir logs\baselines
if not exist checkpoints\baselines mkdir checkpoints\baselines
if not exist results\baselines mkdir results\baselines
set PYCACHE_DIR=analysis\pycache_runtime_cnn_%RANDOM%
if not exist "%PYCACHE_DIR%" mkdir "%PYCACHE_DIR%"

set STATUS=logs\baselines\cnn_baseline_no_redirect_manual.status.txt
set EXITFILE=logs\baselines\cnn_baseline_no_redirect_manual.exit.txt
set PYTHON=.\venv\Scripts\python.exe -E -s -B -X pycache_prefix=%PYCACHE_DIR%

echo START %DATE% %TIME% > "%STATUS%"
echo PYCACHE_DIR %PYCACHE_DIR% >> "%STATUS%"

echo UNET_TRAIN_START %DATE% %TIME% >> "%STATUS%"
%PYTHON% scripts\train_cnn_baseline.py --data_root data --config configs\baselines\unet_polarityfix.yaml --output_dir checkpoints\baselines\unet_polarityfix --device cuda
if errorlevel 1 goto failed
if not exist checkpoints\baselines\unet_polarityfix\best_model.pth (
    echo FAILED %DATE% %TIME% MISSING_UNET_CHECKPOINT checkpoints\baselines\unet_polarityfix\best_model.pth >> "%STATUS%"
    echo 2 > "%EXITFILE%"
    echo UNet training finished without best_model.pth. Please rerun UNet training first.
    pause
    exit /b 2
)

echo UNET_VAL_START %DATE% %TIME% >> "%STATUS%"
%PYTHON% scripts\eval_cnn_baseline.py --data_root data --split val --config configs\baselines\unet_polarityfix.yaml --checkpoint checkpoints\baselines\unet_polarityfix\best_model.pth --output results\baselines\unet_polarityfix_val.json --device cuda
if errorlevel 1 goto failed

echo UNET_TEST_START %DATE% %TIME% >> "%STATUS%"
%PYTHON% scripts\eval_cnn_baseline.py --data_root data --split test --config configs\baselines\unet_polarityfix.yaml --checkpoint checkpoints\baselines\unet_polarityfix\best_model.pth --output results\baselines\unet_polarityfix_test.json --device cuda
if errorlevel 1 goto failed

echo DEEPLAB_TRAIN_START %DATE% %TIME% >> "%STATUS%"
%PYTHON% scripts\train_cnn_baseline.py --data_root data --config configs\baselines\deeplabv3_polarityfix.yaml --output_dir checkpoints\baselines\deeplabv3_polarityfix --device cuda
if errorlevel 1 goto failed
if not exist checkpoints\baselines\deeplabv3_polarityfix\best_model.pth (
    echo FAILED %DATE% %TIME% MISSING_DEEPLAB_CHECKPOINT checkpoints\baselines\deeplabv3_polarityfix\best_model.pth >> "%STATUS%"
    echo 3 > "%EXITFILE%"
    echo DeepLab training finished without best_model.pth. Please rerun DeepLab training first.
    pause
    exit /b 3
)

echo DEEPLAB_VAL_START %DATE% %TIME% >> "%STATUS%"
%PYTHON% scripts\eval_cnn_baseline.py --data_root data --split val --config configs\baselines\deeplabv3_polarityfix.yaml --checkpoint checkpoints\baselines\deeplabv3_polarityfix\best_model.pth --output results\baselines\deeplabv3_polarityfix_val.json --device cuda
if errorlevel 1 goto failed

echo DEEPLAB_TEST_START %DATE% %TIME% >> "%STATUS%"
%PYTHON% scripts\eval_cnn_baseline.py --data_root data --split test --config configs\baselines\deeplabv3_polarityfix.yaml --checkpoint checkpoints\baselines\deeplabv3_polarityfix\best_model.pth --output results\baselines\deeplabv3_polarityfix_test.json --device cuda
if errorlevel 1 goto failed

echo SUCCESS %DATE% %TIME% >> "%STATUS%"
echo 0 > "%EXITFILE%"
echo CNN baseline experiments completed.
exit /b 0

:failed
echo FAILED %DATE% %TIME% ERRORLEVEL=%ERRORLEVEL% >> "%STATUS%"
echo %ERRORLEVEL% > "%EXITFILE%"
echo CNN baseline experiments failed. Check the console output above.
pause
exit /b %ERRORLEVEL%
