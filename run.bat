@echo off
rem One-click rerun: download (if missing) -> clean -> stats -> figures
cd /d "%~dp0"
python -I fintruthqa_pipeline.py
pause
