# NFL Video Timestamp Generator - Installation & Troubleshooting Guide

## Quick Start

1. **Install FFmpeg** (required):
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian)

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install PySceneDetect for better detection**:
   ```bash
   pip install scenedetect
   ```

4. **Test the installation**:
   ```bash
   python nfl_timestamps_working.py --setup
   python nfl_timestamps_working.py --test
   ```

## Common Issues and Solutions

### Issue 1: "ERROR - All detection methods failed"

**Solutions:**
1. Lower the threshold: `--threshold 0.1`
2. Install PySceneDetect: `pip install scenedetect`
3. Check FFmpeg installation: `ffmpeg -version`
4. Try with a smaller video file first

### Issue 2: "could not convert string to float: '15/1'"

This is **FIXED** in this version. The frame rate parsing now handles fractional formats correctly.

### Issue 3: "FFmpeg scene detection timed out"

**Solutions:**
1. Use a shorter video segment for testing
2. The timeout is now 10 minutes (increased from 5)
3. Try PySceneDetect method: program will automatically fall back

### Issue 4: "No valid plays found in game log"

**Solutions:**
1. Check your JSON format against `sample_game_log.json`
2. Ensure required fields: `playId`, `quarter`, `clock`, `description`
3. Use UTF-8 encoding for the JSON file

### Issue 5: Video format not supported

**Supported formats**: MP4, AVI, MOV, MKV
**Convert unsupported formats**:
```bash
ffmpeg -i input.avi output.mp4
```

## Testing with Small Files

Create a test video from your main video:
```bash
# Extract first 5 minutes for testing
ffmpeg -i your_video.mp4 -t 300 test_video.mp4

# Test with the small file
python nfl_timestamps_working.py test_video.mp4 game_log.json --threshold 0.2
```

## Performance Tips

1. **For large videos**: Process in segments
2. **For faster results**: Use `--no-llm` flag
3. **For better accuracy**: Install PySceneDetect
4. **For debugging**: Use `--debug` flag

## Method Priority

The program tries methods in this order:
1. **PySceneDetect** (most reliable, install with `pip install scenedetect`)
2. **FFmpeg scene detection** (corrected syntax)
3. **FFprobe analysis** (corrected frame rate parsing)  
4. **Manual estimation** (NFL pattern-based fallback)

## Verification

After running, check the output:
```json
{
  "metadata": {
    "detection_methods": ["pyscenedetect"],  // Shows which method worked
    "total_plays": 159,
    "detected_scenes": 45,
    "refined_timestamps": 38
  },
  "timestamps": [...]
}
```

## Getting Help

If you're still having issues:
1. Run with `--debug` flag for detailed logs
2. Try with the sample files first: `python nfl_timestamps_working.py --setup`
3. Check that your video plays correctly in media players
4. Verify your JSON structure matches the sample

## System Requirements

**Minimum:**
- Python 3.8+
- FFmpeg installed
- 4GB RAM
- 1GB free disk space

**Recommended:**
- Python 3.10+
- PySceneDetect installed
- 8GB+ RAM
- SSD storage
- LM Studio for AI analysis (optional)
