# NFL Video Timestamp Generator - Troubleshooting Guide

## Common Errors and Solutions

### 1. "FFmpeg scene detection timed out"

**Causes:**
- Video file is very large (>2GB)
- Complex video with many scene changes
- System resources are limited

**Solutions:**
- Try a lower threshold: `--threshold 0.2`
- Process a shorter segment of the video first
- Increase available RAM/CPU resources
- Use the `--fast` mode if available

### 2. "No scene changes detected"

**Causes:**
- Threshold is too high for your video
- Video has very few actual scene changes
- FFmpeg command syntax issues

**Solutions:**
- Lower the threshold: `--threshold 0.1` or `--threshold 0.2`
- Check video format compatibility
- Try PySceneDetect: `pip install scenedetect`
- Use debug mode: add `--debug` flag

### 3. FFmpeg Installation Issues

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH environment variable
- Use Windows Subsystem for Linux (WSL)

**macOS:**
- Install with Homebrew: `brew install ffmpeg`
- Or download from official site

**Linux:**
- Ubuntu/Debian: `sudo apt install ffmpeg`
- CentOS/RHEL: `sudo yum install ffmpeg`

### 4. "No valid plays found in game log"

**Causes:**
- JSON format doesn't match expected structure
- File encoding issues
- Missing required fields

**Solutions:**
- Check JSON format against sample_game_log.json
- Ensure UTF-8 encoding
- Validate JSON syntax
- Check required fields: playId, quarter, clock, description

### 5. Video Format Issues

**Supported formats:**
- MP4 (recommended)
- AVI
- MOV
- MKV
- WMV (with proper codecs)

**If unsupported:**
- Convert with FFmpeg: `ffmpeg -i input.avi output.mp4`
- Use modern video codecs (H.264/H.265)

## Performance Optimization

### Large Video Files
```bash
# Process in segments
python nfl_timestamps_fixed.py video.mp4 game_log.json --max-duration 3600

# Lower quality for faster processing
ffmpeg -i original.mp4 -vf "scale=640:360" -r 15 small.mp4
python nfl_timestamps_fixed.py small.mp4 game_log.json
```

### Faster Detection
```bash
# Skip LLM analysis for speed
python nfl_timestamps_fixed.py video.mp4 game_log.json --no-llm

# Use time-based fallback
python nfl_timestamps_fixed.py video.mp4 game_log.json --method fallback
```

## Debug Mode

Enable detailed logging:
```bash
python nfl_timestamps_fixed.py video.mp4 game_log.json --debug
```

This will show:
- Detection method selection
- FFmpeg command execution
- Scene parsing details
- LLM communication
- Timestamp matching process

## Alternative Tools

If the main script fails, try:

1. **Manual PySceneDetect:**
```bash
scenedetect -i video.mp4 detect-content list-scenes
```

2. **Simple FFmpeg scene detection:**
```bash
ffmpeg -i video.mp4 -filter_complex "select='gt(scene,0.3)',showinfo" -f null - 2>&1 | grep showinfo
```

3. **Frame-by-frame analysis:**
```bash
ffmpeg -i video.mp4 -vf "select='eq(n,0)+gt(scene,0.3)'" frames/%05d.jpg
```

## Getting Help

1. Check the log file for detailed error messages
2. Try with sample video and JSON files first
3. Test individual components (FFmpeg, LM Studio, JSON parsing)
4. Use the debug mode for detailed output
5. Create an issue with system info and error logs

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- FFmpeg installed

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- SSD storage
- LM Studio with local model

## Contact Support

Include this information when reporting issues:
- Operating system and version
- Python version
- FFmpeg version (`ffmpeg -version`)
- Video file format and size
- Complete error message
- JSON structure (sanitized)
