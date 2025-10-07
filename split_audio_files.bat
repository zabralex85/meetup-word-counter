@echo off
echo ========================================
echo Audio File Splitter for OpenAI Whisper
echo ========================================
echo.
echo This script splits large audio files into smaller chunks
echo that can be processed by OpenAI Whisper API (25MB limit).
echo.
echo Features:
echo - Preserves folder structure (radio, tv, tech-meetings)
echo - Creates 20-minute chunks (optimized for size)
echo - Uses MP3 format for smaller file sizes
echo - Maintains audio quality for transcription
echo.
echo Starting audio file splitting...
echo.

python split_audio.py --input-dir demo-files-big --output-dir demo-files-split --max-size 20 --chunk-duration 20

echo.
echo ========================================
echo Splitting completed!
echo ========================================
echo.
echo Split files are now in: demo-files-split\
echo You can now run: run_processing_openai.bat
echo.
echo Note: The split files preserve the original folder structure:
echo - demo-files-split\radio\
echo - demo-files-split\tv\
echo - demo-files-split\tech-meetings\
echo.

pause
