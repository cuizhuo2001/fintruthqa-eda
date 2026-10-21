@echo off
rem One-click rerun of the evaluation: TF-IDF baseline -> BERT baseline -> DeepSeek zero/few-shot -> table + figures
rem LLM outputs are cached in output\eval\llm_cache, so a rerun costs nothing unless the cache is deleted.
cd /d "%~dp0"
python -I eval_tfidf.py || goto :fail
python -I eval_bert.py || echo [skip] BERT failed (needs torch + transformers + GPU), report will leave it out
python -I eval_llm.py || goto :fail
python -I eval_report.py || goto :fail
pause
exit /b 0
:fail
echo Failed, see the error above.
pause
exit /b 1
