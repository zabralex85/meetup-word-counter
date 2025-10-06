@echo off
echo ========================================
echo Hebrew Audio Word Counter - ULTRA LOW CPU MODE
echo ========================================
echo.
echo This mode uses maximum optimization for minimal CPU usage:
echo - Tiny model (fastest, least accurate)
echo - Minimal batch size (4)
echo - No word alignment
echo - Aggressive memory cleanup
echo - CPU-only processing (no GPU)
echo.
echo Demo directory structure found:
echo - demo-files\tv\ (for TV audio files)
echo - demo-files\radio\ (for radio audio files)  
echo - demo-files\tech-meetings\ (for meeting audio files)
echo.
echo Starting audio processing in ULTRA LOW CPU mode...
echo This will be very fast but with reduced accuracy.
echo.

python process_audio.py --model-size tiny --batch-size 4 --device cpu --low-cpu

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
echo Note: Ultra low CPU mode prioritizes speed over accuracy.
echo For better accuracy, use run_processing_low_cpu.bat or run_processing.bat
echo.

pause
