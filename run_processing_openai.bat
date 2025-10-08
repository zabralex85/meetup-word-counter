@echo off
REM Hebrew Audio Word Counter - OpenAI Whisper API
REM This script runs the audio processing using OpenAI's Whisper API

echo ========================================
echo Hebrew Audio Word Counter - OpenAI API
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if the main script exists
if not exist "process_audio_openai.py" (
    echo ERROR: process_audio_openai.py not found
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

REM Check if demo-files-mp3 directory exists
if not exist "demo-files-mp3" (
    echo ERROR: demo-files-mp3 directory not found
    echo Please make sure the audio files are in the demo-files-mp3 directory
    pause
    exit /b 1
)

echo Processing audio files in demo-files-mp3 directory...
echo Using OpenAI Whisper API for transcription
echo.

REM Run the processing script with OpenAI Whisper
python process_audio_openai.py --demo-files demo-files-mp3 --low-cpu

echo.
echo ========================================
echo Processing completed!
echo ========================================
echo.
echo Results saved to:
echo - hebrew_word_analysis_openai.json
echo - hebrew_word_analysis_openai.csv
echo - audio_processing_openai.log
echo.

REM Check if results were created
if exist "hebrew_word_analysis_openai.json" (
    echo ✅ JSON results file created successfully
) else (
    echo ⚠️  JSON results file not found - check logs for errors
)

if exist "hebrew_word_analysis_openai.csv" (
    echo ✅ CSV results file created successfully
) else (
    echo ⚠️  CSV results file not found - check logs for errors
)

echo.
echo Press any key to exit...
pause >nul