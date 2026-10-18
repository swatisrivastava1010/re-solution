@echo off
rem Double-click this file to start Re:Solution on Windows.
rem Anything you add after the file name is passed to the app, for example:
rem   "Start Re-Solution.bat" --no-browser --port 7861
rem The script uses goto instead of ( ) blocks so a folder name with
rem brackets, such as "Re-Solution (1)", can't break it.

setlocal

rem PyTorch, the engine Re:Solution runs on, has no complete version for Windows
rem on Arm yet, so say that plainly instead of failing with a cryptic error. The
rem registry has the PC's own processor, even if this window was opened by a
rem program running under emulation.
reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PROCESSOR_ARCHITECTURE 2>nul | findstr /i "ARM64" >nul
if not errorlevel 1 goto windows_on_arm
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" goto windows_on_arm
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" goto windows_on_arm

rem Work in the app's folder. pushd rather than cd: it also works in a shared
rem network folder (\\computer\share), which it lends a drive letter until popd.
pushd "%~dp0"
rem A lone copy of this file (for example one opened from inside the zip) can't
rem start anything: say what to do instead of failing with a cryptic error.
if not exist "pyproject.toml" goto incomplete_folder

rem The uv installer puts uv here. Only this window's PATH changes: the
rem launcher never edits your Windows settings.
set "UV_BIN_DIR=%USERPROFILE%\.local\bin"
set "PATH=%UV_BIN_DIR%;%PATH%"

rem Which PyTorch build to use: detected each time with the same rules as
rem install.ps1 and install.sh (that's quick), so a folder copied from another
rem computer, or a driver updated since, gets the right one. A "cu130" saved in
rem .torch-variant by install.ps1 or an earlier start is kept when nvidia-smi is
rem there but doesn't answer this time: a driver hiccup shouldn't swap 3 GB of
rem PyTorch for the processor version and back.
set "VARIANT="
if exist ".torch-variant" for /f "usebackq tokens=1" %%v in (".torch-variant") do if not defined VARIANT set "VARIANT=%%v"
set "SAVED=%VARIANT%"
call :detect_variant
if "%SAVED%"=="cu130" if "%NVIDIA%"=="no_answer" set "VARIANT=cu130"
if "%VARIANT%"=="%SAVED%" goto have_variant
rem Remember it for next time; a read-only folder just means detecting again.
>".torch-variant" echo %VARIANT%
:have_variant

where uv >nul 2>nul
if not errorlevel 1 goto start

echo First-time setup: installing uv (a small tool that installs everything else)...
powershell -NoProfile -ExecutionPolicy ByPass -c "$env:UV_NO_MODIFY_PATH=1; $env:UV_INSTALL_DIR=$env:UV_BIN_DIR; irm https://astral.sh/uv/install.ps1 | iex"
where uv >nul 2>nul
if not errorlevel 1 goto start
echo.
echo Couldn't install uv. Check that this PC is connected to the internet, then try again.
pause
popd
exit /b 1

:start
if "%VARIANT%"=="cu130" goto start_cuda
if "%NVIDIA%"=="old_card" goto start_old_card
echo Using the processor (CPU only: slow): no NVIDIA graphics card with driver 580 or newer was found.
goto start_cpu
:start_old_card
echo Using the processor (CPU only: slow): this NVIDIA graphics card is too old for the graphics
echo software Re:Solution uses. It needs a GeForce RTX 20 or GTX 16 series card, or newer.
:start_cpu
echo Starting Re:Solution. The first start downloads about 400 MB and can take a few minutes.
goto run
:start_cuda
echo Starting Re:Solution with your NVIDIA graphics card. The first start downloads about 3 GB and can take 10 minutes or more.
:run
echo Keep this window open while you use Re:Solution. Close it to stop.
echo.

rem --frozen installs exactly the versions in uv.lock; --no-dev skips the test
rem tools; --extra picks the PyTorch build.
uv run --frozen --no-dev --extra %VARIANT% python -m upscaler serve %*
set "STATUS=%ERRORLEVEL%"
popd

rem -1073741510 (0xC000013A) means someone pressed Ctrl+C, a normal way to stop.
if "%STATUS%"=="0" exit /b 0
if "%STATUS%"=="-1073741510" exit /b 0
echo.
echo Something went wrong.
pause
exit /b %STATUS%

:incomplete_folder
popd
echo Sorry, Re:Solution can't start from here: the rest of its folder is missing.
echo If you opened it from the zip file, right-click the zip, choose "Extract All",
echo then open the new folder and start Re:Solution from there.
echo.
pause
exit /b 1

:windows_on_arm
echo Sorry, Re:Solution needs a Windows PC with an Intel or AMD processor.
echo PyTorch, the engine it runs on, has no complete version for Windows on Arm yet.
echo.
pause
exit /b 1

rem Sets VARIANT to "cu130" when this PC has an NVIDIA graphics card that the CUDA
rem 13 build supports, otherwise to "cpu". NVIDIA says why:
rem   none        no NVIDIA driver (so no NVIDIA card to use)
rem   no_answer   nvidia-smi is there but didn't answer sensibly this time
rem   old_driver  a driver older than 580, which the CUDA 13 build needs
rem   old_card    a card older than the GeForce RTX 20 / GTX 16 series (compute
rem               capability 7.5), which the CUDA 13 build has no code for. Driver
rem               580 is the last one for GTX 10 cards, so they report it too.
rem   ok          the cu130 build will work
:detect_variant
set "VARIANT=cpu"
set "NVIDIA=none"
set "DRIVER="
set "CAP="
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :eof
set "NVIDIA=no_answer"
for /f "tokens=1 delims=. " %%d in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do if not defined DRIVER set "DRIVER=%%d"
if not defined DRIVER goto :eof
rem Only a plain number counts: anything else is an error message, meaning no usable driver.
echo %DRIVER%| findstr /r /x "[0-9][0-9]*" >nul
if errorlevel 1 goto :eof
set "NVIDIA=old_driver"
if %DRIVER% LSS 580 goto :eof
rem The first card's compute capability, for example 8.6, read as 86. Drivers
rem older than 510 don't know this question, so it is only asked now.
set "NVIDIA=no_answer"
for /f "tokens=1,2 delims=. " %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do if not defined CAP set "CAP=%%a%%b"
if not defined CAP goto :eof
echo %CAP%| findstr /r /x "[0-9][0-9]*" >nul
if errorlevel 1 goto :eof
set "NVIDIA=old_card"
if %CAP% LSS 75 goto :eof
set "NVIDIA=ok"
set "VARIANT=cu130"
goto :eof
