@echo off
echo ========================================
echo Hebrew Audio Word Counter - OpenAI Whisper
echo ========================================
echo.
echo This version uses OpenAI's Whisper API for reliable transcription.
echo Make sure you have set your OPENAI_API_KEY environment variable.
echo.
echo Demo directory structure found:
echo - demo-files\tv\ (for TV audio files)
echo - demo-files\radio\ (for radio audio files)  
echo - demo-files\tech-meetings\ (for meeting audio files)
echo.
echo Supported formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM
echo Maximum file size: 25MB per file
echo.
echo Starting audio processing with OpenAI Whisper...
echo This will use your OpenAI API credits.
echo.

python process_audio_openai.py

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
echo Check the log file for any errors or cost estimates.
echo.

pause
