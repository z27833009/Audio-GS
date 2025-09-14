@echo off
echo ========================================
echo Audio-GS Training Script
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist samples mkdir samples
if not exist logs mkdir logs

REM 检查是否有音频文件
set AUDIO_FILE=""
if exist samples\*.wav (
    for %%f in (samples\*.wav) do (
        set AUDIO_FILE=%%f
        goto :found
    )
)
if exist samples\*.mp3 (
    for %%f in (samples\*.mp3) do (
        set AUDIO_FILE=%%f
        goto :found
    )
)

:found
if %AUDIO_FILE%=="" (
    echo [INFO] No audio files found in samples/
    echo [INFO] Creating test audio...
    python -c "import train_example; train_example.create_test_audio()"
    set AUDIO_FILE=samples\test_chord.wav
)

echo.
echo Training with audio file: %AUDIO_FILE%
echo.
echo Select training mode:
echo 1. Quick test (100 Gaussians, 1000 steps)
echo 2. Standard quality (500 Gaussians, 5000 steps)
echo 3. High quality (1000 Gaussians, 10000 steps)
echo 4. Custom settings
echo.

set /p MODE="Enter choice (1-4): "

if "%MODE%"=="1" (
    set NUM_GAUSSIANS=100
    set NUM_STEPS=1000
    set CONFIG=default
) else if "%MODE%"=="2" (
    set NUM_GAUSSIANS=500
    set NUM_STEPS=5000
    set CONFIG=default
) else if "%MODE%"=="3" (
    set NUM_GAUSSIANS=1000
    set NUM_STEPS=10000
    set CONFIG=music
) else if "%MODE%"=="4" (
    set /p NUM_GAUSSIANS="Enter number of Gaussians (e.g., 500): "
    set /p NUM_STEPS="Enter number of steps (e.g., 5000): "
    set CONFIG=default
) else (
    echo Invalid choice. Using default settings.
    set NUM_GAUSSIANS=500
    set NUM_STEPS=5000
    set CONFIG=default
)

echo.
echo ========================================
echo Configuration:
echo   Audio: %AUDIO_FILE%
echo   Gaussians: %NUM_GAUSSIANS%
echo   Steps: %NUM_STEPS%
echo   Config: configs/%CONFIG%.yaml
echo ========================================
echo.

REM 运行训练
python main.py ^
    --input_path %AUDIO_FILE% ^
    --num_gaussians %NUM_GAUSSIANS% ^
    --num_steps %NUM_STEPS% ^
    --config configs/%CONFIG%.yaml

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Training completed successfully!
    echo Check the logs/ directory for results.
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Training failed. Check error messages above.
    echo ========================================
)

echo.
pause