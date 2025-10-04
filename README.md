# NFL AI Timestamps

An AI-powered tool that automatically detects and timestamps NFL plays in game videos by matching visual and audio content with play-by-play data using local LLM (LM Studio).

## Features

- **Scene Detection**: Automatically detects scene changes in NFL game videos
- **Audio Transcription**: Uses OpenAI Whisper to transcribe commentary and on-field audio
- **Visual Analysis**: Extracts and analyzes on-screen text (scores, down/distance, time, etc.) using vision-capable LLMs
- **Play Matching**: Intelligently matches video scenes with play-by-play data from game logs
- **Local LLM Integration**: Uses LM Studio for privacy-preserving AI analysis
- **Efficient Processing**: Transcribes video once and reuses transcription for faster processing

## Prerequisites

- **Python 3.8+**
- **LM Studio** running locally with a loaded model (default: http://localhost:1234)
- **FFmpeg** (required by moviepy for video processing)

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd "NFL AI Timestamps"
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
```

3. **Activate the virtual environment:**

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Set up LM Studio:**
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Load a model (vision-capable models recommended for best results)
   - Start the local server (default: http://localhost:1234)

## Usage

### Basic Usage

```bash
python nfl_processor.py <video_file> <game_log_file>
```

**Example:**
```bash
python nfl_processor.py game_videos/game.mp4 game_logs/game_log.json
```

### Command Line Options

- `video_file` - Path to the NFL game video file (required)
- `game_log` - Path to the game log JSON file with play-by-play data (required)
- `--lm-studio-url URL` - LM Studio API URL (default: http://localhost:1234)
- `--debug-scenes` - Output detected scenes to JSON for debugging (skips LLM processing)
- `--no-short` - When used with --debug-scenes, only include scenes ≥10 seconds
- `--transcription FILE` - Path to existing transcription JSON file (skips re-transcribing)

### Advanced Usage Examples

**Debug mode to analyze scene detection:**
```bash
python nfl_processor.py game_videos/game.mp4 game_logs/game_log.json --debug-scenes
```

**Skip short scenes in debug output:**
```bash
python nfl_processor.py game_videos/game.mp4 game_logs/game_log.json --debug-scenes --no-short
```

**Reuse existing transcription:**
```bash
python nfl_processor.py game_videos/game.mp4 game_logs/game_log.json --transcription transcriptions/transcription_game_20250923_123841.json
```

**Custom LM Studio URL:**
```bash
python nfl_processor.py game_videos/game.mp4 game_logs/game_log.json --lm-studio-url http://localhost:8080
```

## Input Requirements

### Game Log JSON Format

The game log file should contain play-by-play data in JSON format with the following structure:

```json
[
  {
    "play_id": "...",
    "game_id": "...",
    "home_team": "...",
    "away_team": "...",
    "week": "...",
    "posteam": "...",
    "defteam": "...",
    "qtr": "...",
    "down": "...",
    "time": "...",
    "yrdln": "...",
    "ydstogo": "...",
    "desc": "...",
    "play_type": "...",
    "yards_gained": "...",
    "passer_player_name": "...",
    "receiver_player_name": "...",
    "rusher_player_name": "...",
    ...
  }
]
```

### Video Format

Supports common video formats (MP4, MKV, AVI, etc.) that can be processed by OpenCV and moviepy.

## Output

The tool generates three types of output files in the `processing_results/` directory:

1. **results_[timestamp].json** - Matched plays with timestamps
   ```json
   {
     "game": "2023 Week 1 - Away @ Home",
     "videoUrl": "path/to/video.mp4",
     "clips": [
       {
         "playIndex": 0,
         "startTime": 123.45,
         "endTime": 145.67,
         "playDescription": "...",
         "play": { ... },
         "matches": [ ... ],
         "total_confidence": 0.85
       }
     ]
   }
   ```

2. **llm_responses_[timestamp].json** - All LLM interactions for debugging

3. **debug_scenes_[timestamp].json** - Scene detection output (when using --debug-scenes)

Transcription files are saved to the `transcriptions/` directory.

## How It Works

1. **Video Transcription**: The entire video is transcribed once using Whisper (medium model)
2. **Scene Detection**: Detects scene changes using PySceneDetect with content detection
3. **Scene Analysis**: For each scene ≥15 seconds:
   - Extracts relevant audio transcription from cached results
   - Analyzes video frames to extract on-screen text
   - Verifies if the scene shows an actual football play
4. **Play Matching**: Matches extracted data with game log plays based on:
   - Play descriptions
   - Player names
   - Down/distance
   - Time/score information
5. **Confidence Scoring**: Ranks matches and keeps only the highest confidence match per play

## Troubleshooting

**"Error loading Whisper model"**
- Ensure you have sufficient disk space and RAM
- The medium model requires ~5GB of space and ~2GB RAM

**"LM Studio API error"**
- Verify LM Studio is running and the server is started
- Check the API URL matches your LM Studio configuration
- Ensure a model is loaded in LM Studio

**"Scene detection failed"**
- Verify FFmpeg is properly installed
- Check video file is not corrupted
- Ensure video file format is supported

**Slow processing**
- Use `--transcription` flag to reuse existing transcriptions
- Consider using a smaller/faster LLM model in LM Studio
- Reduce video resolution or length for testing

## Performance

- Scene detection: ~30-60 seconds for a typical game video
- Full transcription: ~5-15 minutes (one-time per video)
- Scene processing: ~10-15 seconds per scene (with LLM calls)
- Total processing time: Varies based on video length and number of scenes

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]
