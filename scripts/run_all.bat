@echo off
echo ==============================================================================
echo PYRORL: FULL EVALUATION AND VISUALIZATION PIPELINE
echo ==============================================================================
echo.

if not exist checkpoints\ppo_california_best.zip (
    echo ERROR: California model missing
    exit /b 1
)
if not exist checkpoints\ppo_saudi_best.zip (
    echo ERROR: Saudi model missing
    exit /b 1
)

:: Generate safe timestamp for directory
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,4%_%datetime:~4,2%_%datetime:~6,2%_%datetime:~8,2%_%datetime:~10,2%
set OUTPUT_DIR=logs\exp_%TIMESTAMP%

echo [INFO] Output directory: %OUTPUT_DIR%

echo.
echo ==============================
echo Running Step 1/3: Evaluation...
echo ==============================
call python scripts/run_scenario_matrix.py ^
    --ca_model checkpoints/ppo_california_best.zip ^
    --sa_model checkpoints/ppo_saudi_best.zip ^
    --episodes 50 ^
    --seed 42 ^
    --output_dir %OUTPUT_DIR%

if %errorlevel% neq 0 (
    echo ERROR: Step 1 failed. Exiting pipeline.
    exit /b %errorlevel%
)

echo.
echo ==============================
echo Running Step 2/3: Plotting...
echo ==============================
call python scripts/plot_scenario_results.py --output_dir %OUTPUT_DIR%

if %errorlevel% neq 0 (
    echo ERROR: Step 2 failed. Exiting pipeline.
    exit /b %errorlevel%
)

echo.
echo ==============================
echo Running Step 3/3: Visualization...
echo ==============================
call python scripts/generate_all_visualizations.py ^
    --ca_model checkpoints/ppo_california_best.zip ^
    --sa_model checkpoints/ppo_saudi_best.zip ^
    --output_dir %OUTPUT_DIR% ^
    --annotate ^
    --seed 42

if %errorlevel% neq 0 (
    echo ERROR: Step 3 failed. Exiting pipeline.
    exit /b %errorlevel%
)

echo.
echo ======================================
echo PIPELINE COMPLETED SUCCESSFULLY
echo Output: %OUTPUT_DIR%
echo ======================================
