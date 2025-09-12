# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL Video Timestamp Generator that analyzes NFL broadcast videos to automatically detect and timestamp key plays using AI-powered scene detection. The system correlates detected scene changes with game log data to identify when specific plays occurred in video footage.

## Architecture

The project consists of two main Python scripts:

### Core Components

1. **nfl_timestamps.py** - Main timestamp generation system with multiple classes:
   - `LMStudioClient` - Interfaces with local LLM server for scene analysis
   - `WorkingSceneDetector` - Detects scene changes using PySceneDetect, FFmpeg, or fallback methods
   - `GameLogProcessor` - Processes JSON game log files containing play-by-play data
   - `TimestampRefiner` - Correlates detected scenes with game log plays to generate final timestamps

2. **getInfo.py** - Video analysis utility that extracts first frames and uses Ollama VLMs to extract text content

### Data Structures

- `PlayData` - Individual play information (game time, quarter, down, distance, description)
- `DetectedScene` - Scene change data (timestamp, confidence, frame difference)  
- `RefinedTimestamp` - Final output correlating video timestamps with play data

### Detection Methods (in priority order)

1. **PySceneDetect** - Most reliable, requires `pip install scenedetect`
2. **FFmpeg scene detection** - Built-in fallback using video analysis
3. **FFprobe analysis** - Frame rate and metadata analysis
4. **Manual estimation** - NFL pattern-based timing fallback

## Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Optional but recommended for better detection
pip install scenedetect

# Create sample files for testing
python nfl_timestamps.py --setup
```

### Running the System
```bash
# Basic usage
python nfl_timestamps.py video.mp4 game_log.json

# With custom settings and debug output
python nfl_timestamps.py video.mp4 game_log.json --output results.json --threshold 0.3 --debug

# Process video with Ollama text extraction
python getInfo.py
```

### Testing and Validation
```bash
# Test with sample files
python nfl_timestamps.py --setup
python nfl_timestamps.py --test

# Test individual components
ffmpeg -version  # Verify FFmpeg installation
python -c "import scenedetect; print('PySceneDetect available')"
```

## Key File Locations

- `game_logs/` - Contains NFL game log JSON files with play-by-play data
- `game_videos/` - Contains NFL broadcast video files (.mp4)
- `sample_game_log.json` - Example JSON structure for game logs
- `timestamps.json` - Output file with detected timestamps and correlations

## JSON Data Format

Game logs must contain a "plays" array with required fields:
- `playId` - Unique identifier
- `quarter` - Quarter number (1-4)  
- `clock` - Game time in MM:SS format
- `down` - Down number (1-4)
- `distance` - Yards to go
- `playType` - Type of play (rush, pass, etc.)
- `description` - Play description

## Dependencies

### Required
- Python 3.8+
- FFmpeg (must be installed and in PATH)
- numpy>=1.21.0
- requests>=2.25.0

### Optional but Recommended
- scenedetect>=0.6.0 - Significantly improves scene detection accuracy
- LM Studio with local model - For AI-powered scene analysis
- Ollama - For video frame text extraction (getInfo.py)

### Computer Vision (Alternative)
- opencv-python>=4.5.0
- imageio-ffmpeg>=0.4.0

## Common Development Patterns

### Error Handling
- All detection methods have fallback chains to ensure some output is always generated
- Timeout handling for long-running FFmpeg operations (10 minute timeout)
- Graceful degradation when optional dependencies are missing

### Configuration
- Threshold adjustment (0.1-0.8) for scene detection sensitivity
- Multiple output formats and confidence scoring
- Debug mode for detailed logging and troubleshooting

### Performance Optimization
- Process large videos in segments using `--max-duration`
- Use `--no-llm` flag for faster processing without AI analysis
- Lower video resolution for faster testing: `ffmpeg -i original.mp4 -vf "scale=640:360" -r 15 small.mp4`