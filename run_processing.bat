@echo off
echo ========================================
echo Hebrew Audio Word Counter
echo ========================================
echo.

echo Demo directory structure found:
echo - demo-files\tv\ (for TV audio files)
echo - demo-files\radio\ (for radio audio files)  
echo - demo-files\tech-meetings\ (for meeting audio files)
echo.

echo Starting audio processing...
echo This may take a while depending on the number and size of audio files.
echo.

python process_audio.py --model-size large-v2 --batch-size 16

echo.
echo ========================================
echo Processing completed!
echo ========================================
echo.
echo Results saved to:
echo - hebrew_word_analysis.json
echo - hebrew_word_analysis.csv
echo - audio_processing.log
echo.

pause