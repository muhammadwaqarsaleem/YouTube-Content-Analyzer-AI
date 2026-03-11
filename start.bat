@echo off
title YouTube Video Analysis System Launcher
echo =======================================================
echo Starting all servers (Backend, Frontend, Deno)...
echo =======================================================
echo.
python start_server.py --frontend
pause
