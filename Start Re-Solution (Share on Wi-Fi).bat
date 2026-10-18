@echo off
rem Double-click this file to start Re:Solution and let phones and computers on
rem your Wi-Fi use it too. Everything else is the same as "Start Re-Solution.bat".

rem A lone copy of this file (for example one opened from inside the zip) can't
rem start anything: say what to do instead of closing the window at once.
if not exist "%~dp0Start Re-Solution.bat" goto incomplete_folder

echo Sharing on Wi-Fi: phones and computers on the same Wi-Fi can open the address shown below.
echo If Windows asks about network access, tick "Private networks" and click "Allow access".
echo.

call "%~dp0Start Re-Solution.bat" --share %*
exit /b %ERRORLEVEL%

:incomplete_folder
echo Sorry, Re:Solution can't start from here: the rest of its folder is missing.
echo If you opened it from the zip file, right-click the zip, choose "Extract All",
echo then open the new folder and start Re:Solution from there.
echo.
pause
exit /b 1
