@echo off
setlocal enabledelayedexpansion
REM run_all_experiments.bat
REM Run experiments for MuProMAC scenarios with XGBoost and single bucketing
REM Testing all encoding methods: agg, laststate, index, combined
REM Place this file in the same directory as experiments_param_optim.py and experiments_final.py

echo ====================================================================
echo Running experiments for all MuProMAC scenarios
echo Bucketing: single only
echo Classifier: xgb only
echo Encodings: agg, laststate, index, combined
echo ====================================================================
echo.

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Add parent directory to Python path so it can find transformers/ and bucketers/
set PYTHONPATH=%CD%;%CD%\..;%PYTHONPATH%

REM Define method combinations to test
REM Format: bucket_method cls_encoding cls_method
set methods[0]=single agg xgb
set methods[1]=single laststate xgb
set methods[2]=single index xgb
set methods[3]=single combined xgb

REM Count total methods
set num_methods=1

REM Counter for processed scenarios
set dataset_count=0

REM Get list of datasets from dataset_confs.py
python -c "import dataset_confs; datasets = [k for k in dataset_confs.filename.keys() if k.startswith('mup_')]; print('\n'.join(datasets))" > temp_datasets.txt

REM Loop through each dataset
for /f "tokens=*" %%d in (temp_datasets.txt) do (
    set /a dataset_count+=1
    set "dataset=%%d"
    
    echo.
    echo ====================================================================
    echo [!dataset_count!] Processing dataset: !dataset!
    echo ====================================================================
    echo.
    
    REM Loop through each method combination
    for /L %%i in (0,1,0) do (
        REM Parse method combination
        for /f "tokens=1,2,3" %%a in ("!methods[%%i]!") do (
            set bucket_method=%%a
            set cls_encoding=%%b
            set cls_method=%%c
            
            echo.
            echo ----------------------------------------------------------------
            echo Method: !bucket_method! + !cls_encoding! + !cls_method!
            echo ----------------------------------------------------------------
            
            REM Run hyperparameter optimization
            echo Step 1: Running hyperparameter optimization...
            python experiments_param_optim.py "!dataset!" !bucket_method! !cls_encoding! !cls_method!
            if errorlevel 1 (
                echo ERROR: Hyperparameter optimization failed
                echo Continuing to next method...
            ) else (
                REM Extract best parameters
                echo Step 2: Extracting best parameters...
                python extract_best_params.py
                
                REM Run final evaluation
                echo Step 3: Running final evaluation...
                python experiments_final.py "!dataset!" !bucket_method! !cls_encoding! !cls_method!
                if errorlevel 1 (
                    echo ERROR: Final evaluation failed
                )
            )
        )
    )
    
    echo.
    echo Completed all methods for: !dataset!
)

REM Cleanup
del temp_datasets.txt

echo.
echo ====================================================================
echo Finished processing !dataset_count! datasets with !num_methods! encoding methods
echo ====================================================================

pause