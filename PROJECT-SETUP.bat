@echo off

REM Function to find the correct Python binary
setlocal enabledelayedexpansion

set "required_version=3.9"
set "max_version=4.0"

for /f "tokens=*" %%I in ('where python*.exe') do (
    for /f "tokens=*" %%J in ('%%I -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"') do (
        set "python_version=%%J"
        for /f "tokens=1,2 delims=." %%K in ("!python_version!") do (
            if %%K GEQ 3 (
                if %%K LEQ 3 (
                    if %%L GEQ 9 (
                        if %%L LEQ 11 (
                            if "!python_version!" LSS "!max_version!" (
                                set "PYTHON=%%I"
                                goto :found_python
                            )
                        )
                    )
                )
            )
        )
    )
)

echo You must have a Python version higher than 3.9 and lower than 4.0 to launch this project.
exit /b 1

:found_python
echo Using Python binary: %PYTHON%

REM Display the platform machine
for /f "tokens=*" %%I in ('%PYTHON% -c "import platform; print(platform.machine())"') do (
    set "platform_machine=%%I"
)
echo Platform machine: %platform_machine%

REM Clone the repository in the current directory
git clone https://github.com/nickpadd/EuropeanFootballLeaguePredictor

REM Browse the project directory
cd EuropeanFootballLeaguePredictor || exit /b 1

REM Create a virtual environment
%PYTHON% -m venv EFLPvenv

REM Activate the virtual environment
call EFLPvenv\Scripts\activate.bat

REM Install dependencies for the project using the pip associated with the virtual environment
EFLPvenv\Scripts\pip install -r requirements.txt

echo The project has been successfully installed. Perform the following README steps.