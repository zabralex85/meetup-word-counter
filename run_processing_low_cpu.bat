@echo off
echo ========================================
echo Hebrew Audio Word Counter - LOW CPU MODE
echo ========================================
echo.
echo This mode uses optimized settings for lower CPU usage:
echo - Smaller model (base instead of large-v2)
echo - Reduced batch size (8 instead of 16)
echo - Skipped word alignment for faster processing
echo - Aggressive memory cleanup
echo.
echo Demo directory structure found:
echo - demo-files\tv\ (for TV audio files)
echo - demo-files\radio\ (for radio audio files)  
echo - demo-files\tech-meetings\ (for meeting audio files)
echo.
echo Starting audio processing in LOW CPU mode...
echo This will be faster but may have slightly lower accuracy.
echo.

python process_audio.py --model-size base --batch-size 8 --low-cpu

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
echo Note: Low CPU mode sacrifices some accuracy for speed.
echo For best accuracy, use run_processing.bat instead.
echo.

pause
