# CPU Optimization Guide for Hebrew Audio Processing

This guide explains how to reduce CPU usage when processing Hebrew audio files with the meetup-word-counter tool.

## Available Processing Modes

### 1. Standard Mode (Default)
**File:** `run_processing.bat`
- Model: `large-v2` (highest accuracy)
- Batch size: 16
- Word alignment: Enabled
- **CPU Usage:** High
- **Accuracy:** Highest

### 2. Low CPU Mode
**File:** `run_processing_low_cpu.bat`
- Model: `base` (good balance)
- Batch size: 8
- Word alignment: Disabled
- Memory cleanup: Aggressive
- **CPU Usage:** ~50% reduction
- **Accuracy:** Good (slight reduction)

### 3. Ultra Low CPU Mode
**File:** `run_processing_ultra_low_cpu.bat`
- Model: `tiny` (fastest)
- Batch size: 4
- Device: CPU only
- Word alignment: Disabled
- Memory cleanup: Aggressive
- **CPU Usage:** ~75% reduction
- **Accuracy:** Basic (noticeable reduction)

## Manual Command Line Options

You can also run the script manually with custom settings:

```bash
# Low CPU mode with custom model
python process_audio.py --model-size base --batch-size 8 --low-cpu

# Ultra low CPU mode
python process_audio.py --model-size tiny --batch-size 4 --device cpu --low-cpu

# Custom settings
python process_audio.py --model-size small --batch-size 6 --low-cpu --demo-files demo-files-big
```

## Model Size Comparison

| Model | Size | Speed | Accuracy | CPU Usage |
|-------|------|-------|----------|-----------|
| tiny | ~39 MB | Fastest | Basic | Lowest |
| base | ~74 MB | Fast | Good | Low |
| small | ~244 MB | Medium | Better | Medium |
| medium | ~769 MB | Slow | Good | High |
| large | ~1550 MB | Slower | Very Good | Higher |
| large-v2 | ~1550 MB | Slower | Excellent | Highest |
| large-v3 | ~1550 MB | Slowest | Best | Highest |

## CPU Optimization Features

### 1. Low CPU Mode (`--low-cpu`)
- Uses `int8` compute type instead of `float16`
- Reduces batch size automatically
- Skips word-level alignment
- Aggressive memory cleanup after each file
- Pre-compiled regex patterns for faster text processing

### 2. Memory Management
- Automatic garbage collection after each file
- GPU memory clearing when available
- Immediate audio data cleanup
- Reduced memory footprint

### 3. Processing Optimizations
- Smaller batch sizes for lower memory usage
- Skipped alignment for faster processing
- Pre-compiled regex patterns
- Optimized text processing pipeline

## Performance Recommendations

### For High-End Systems
- Use standard mode for best accuracy
- Consider `large-v3` model for maximum quality

### For Mid-Range Systems
- Use low CPU mode with `base` or `small` model
- Good balance of speed and accuracy

### For Low-End Systems
- Use ultra low CPU mode with `tiny` model
- Accept reduced accuracy for faster processing

### For Batch Processing
- Use low CPU mode to process more files
- Consider processing files in smaller batches

## Monitoring CPU Usage

You can monitor CPU usage during processing:

**Windows:**
```cmd
# Open Task Manager or use PowerShell
Get-Process python | Select-Object ProcessName, CPU, WorkingSet
```

**Linux/Mac:**
```bash
# Monitor CPU usage
top -p $(pgrep -f process_audio.py)
```

## Troubleshooting

### High CPU Usage Issues
1. Use `--low-cpu` flag
2. Reduce batch size further (e.g., `--batch-size 4`)
3. Use smaller model (e.g., `--model-size tiny`)
4. Force CPU-only processing (`--device cpu`)

### Memory Issues
1. Enable low CPU mode
2. Process files in smaller batches
3. Close other applications
4. Use smaller model size

### Slow Processing
1. Use GPU if available (remove `--device cpu`)
2. Increase batch size (if memory allows)
3. Use larger model for better accuracy
4. Disable low CPU mode if system can handle it

## Expected Performance Improvements

| Mode | CPU Reduction | Speed Improvement | Accuracy Loss |
|------|---------------|-------------------|---------------|
| Low CPU | ~50% | ~2x faster | ~5-10% |
| Ultra Low CPU | ~75% | ~4x faster | ~15-25% |

## Best Practices

1. **Start with low CPU mode** - Good balance for most systems
2. **Test with small files first** - Verify settings work for your system
3. **Monitor system resources** - Adjust settings based on performance
4. **Use appropriate model size** - Match model to your accuracy needs
5. **Process in batches** - For large datasets, process in smaller groups

## Command Examples

```bash
# Quick test with low CPU mode
python process_audio.py --model-size tiny --batch-size 4 --low-cpu

# Balanced processing
python process_audio.py --model-size base --batch-size 8 --low-cpu

# Maximum accuracy (if system can handle it)
python process_audio.py --model-size large-v2 --batch-size 16

# Process specific directory
python process_audio.py --demo-files demo-files-big --model-size small --low-cpu
```
