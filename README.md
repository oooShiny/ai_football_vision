# NFL Video Timestamp Generator - Usage Guide

## Installation

1. Install Python 3.8+ 
2. Install FFmpeg: https://ffmpeg.org/download.html
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## LM Studio Setup (Optional but Recommended)

1. Download LM Studio: https://lmstudio.ai/
2. Download a local model (recommended: Llama-3.2-3B or similar)
3. Start the local server in LM Studio
4. Ensure it's running on http://localhost:1234

## Usage

### Basic Usage
```bash
python nfl_timestamps.py video.mp4 game_log.json
```

### With Custom Settings
```bash
python nfl_timestamps.py video.mp4 game_log.json --output results.json --threshold 0.3
```

### Create Sample Files
```bash
python nfl_timestamps.py --setup
```

## JSON Format

Your game log JSON should contain a "plays" array with the following fields:
- playId: Unique identifier
- quarter: Quarter number (1-4)
- clock: Game time (MM:SS format)
- down: Down number (1-4)
- distance: Yards to go
- playType: Type of play (rush, pass, etc.)
- description: Play description

## Output

The program generates a JSON file with:
- Metadata about the analysis
- Array of refined timestamps with play correlations
- Confidence scores and detection methods

## Troubleshooting

- If FFmpeg is not found, install it and add to PATH
- If LM Studio connection fails, the program uses rule-based fallback
- Adjust --threshold (0.1-0.8) if too many/few scenes are detected
- Check JSON format if no plays are loaded

## Tips for Best Results

1. Use high-quality video files
2. Ensure game log times are accurate
3. Start LM Studio before running analysis
4. Experiment with threshold values
5. Review output confidence scores
