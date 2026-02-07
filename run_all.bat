@echo off
REM Full diarization run — processes all unprocessed episodes.
REM Already-processed episodes are skipped automatically.
REM Safe to interrupt and re-run (resumes where it left off).

cd /d "%~dp0"

echo === Diarization: full run ===
echo Start: %date% %time%
echo.

docker compose run --rm whisperx process --workers 1

echo.
echo === Done: %date% %time% ===
