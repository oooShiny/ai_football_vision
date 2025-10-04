#!/usr/bin/env python3
"""
NFL Video Timestamp Processor

A clean, streamlined implementation for processing NFL videos to detect
scene changes and match them with play data using LM Studio LLM.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import requests
import whisper
from moviepy import VideoFileClip
from PIL import Image
import base64


class NFLProcessor:
    """Main processor for NFL video timestamp detection."""

    def __init__(self, video_path: str, game_log_path: str, lm_studio_url: str = "http://localhost:1234", debug_scenes: bool = False, no_short: bool = False, transcription_file: str = None):
        self.video_path = Path(video_path)
        self.game_log_path = Path(game_log_path)
        self.lm_studio_url = lm_studio_url
        self.debug_scenes = debug_scenes
        self.no_short = no_short
        self.transcription_file = transcription_file

        # Set up logging
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_logging()

        # Load game log data
        self.game_log = self.load_game_log()

        # Results storage
        self.results = []  # Will store final unique play matches
        self.play_matches = {}  # Track matches by play_id for uniqueness
        self.llm_responses = []

        # Initialize Whisper model
        self.logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("medium")

        # Cache for full video transcription
        self.full_transcription = None
        self.transcription_segments = []

    def setup_logging(self):
        """Set up logging for this processing run."""
        # Create directories for output files
        Path("processing_results").mkdir(exist_ok=True)
        Path("transcriptions").mkdir(exist_ok=True)

        # Console-only logging for better real-time feedback
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',  # Simpler format for console readability
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Starting NFL processing run {self.run_id}")

    def load_game_log(self) -> List[Dict]:
        """Load and parse the game log JSON file."""
        try:
            with open(self.game_log_path, 'r') as f:
                game_log = json.load(f)
            self.logger.info(f"Loaded game log with {len(game_log)} plays")
            return game_log
        except Exception as e:
            self.logger.error(f"Error loading game log: {e}")
            sys.exit(1)

    def detect_scene_changes(self) -> List[Tuple[float, float]]:
        """Detect scene changes in the video using scenedetect."""
        scene_start = time.time()

        try:
            from scenedetect import detect, ContentDetector

            # Use the modern scenedetect API
            scene_list = detect(str(self.video_path), ContentDetector(threshold=30.0))

            scenes = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
            scene_time = time.time() - scene_start
            self.logger.info(f"Scene detection completed in {scene_time:.1f}s")
            return scenes

        except Exception as e:
            self.logger.error(f"Error detecting scenes: {e}")
            return []

    def call_lm_studio(self, prompt: str, system_message: str = "") -> Optional[str]:
        """Make a text-only request to LM Studio API."""
        try:
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_message} if system_message else {},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }

            # Remove empty system message
            payload["messages"] = [msg for msg in payload["messages"] if msg.get("content")]

            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Log the LLM response
                self.llm_responses.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "text",
                    "prompt": prompt,
                    "system_message": system_message,
                    "response": content
                })

                return content
            else:
                self.logger.error(f"LM Studio API error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error calling LM Studio: {e}")
            return None

    def transcribe_full_video(self) -> bool:
        """Transcribe the entire video once using Whisper, or load from existing file."""
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"STARTING FULL VIDEO TRANSCRIPTION")
            self.logger.info(f"{'='*60}")

            # Check if transcription file is provided
            if self.transcription_file:
                self.logger.info(f"📁 Loading existing transcription from: {self.transcription_file}")
                try:
                    with open(self.transcription_file, 'r') as f:
                        transcription_data = json.load(f)

                    self.full_transcription = transcription_data["full_transcription"]
                    self.transcription_segments = transcription_data["segments"]

                    self.logger.info(f"✅ Transcription loaded successfully!")
                    self.logger.info(f"   📊 Found {len(self.transcription_segments)} speech segments")
                    self.logger.info(f"   📝 Total transcript length: {len(self.full_transcription)} characters")
                    self.logger.info(f"{'='*60}")
                    return True

                except Exception as e:
                    self.logger.error(f"❌ Error loading transcription file: {e}")
                    self.logger.info("   Falling back to fresh transcription...")

            # Generate transcription filename based on video file
            video_name = self.video_path.stem
            transcription_filename = f"transcription_{video_name}_{self.run_id}.json"
            transcription_path = Path("transcriptions") / transcription_filename

            self.logger.info("📼 Extracting audio from full video...")

            # Extract audio from entire video
            video = VideoFileClip(str(self.video_path))
            audio_clip = video.audio

            # Create temporary file for full audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_clip.write_audiofile(temp_audio.name, logger=None)
                temp_audio_path = temp_audio.name

            # Clean up video clips
            audio_clip.close()
            video.close()

            self.logger.info("🎙️  Transcribing full video with Whisper...")
            self.logger.info("   ⏳ This may take several minutes depending on video length")
            transcription_start = time.time()

            # Transcribe with Whisper - get segments for timestamp mapping
            result = self.whisper_model.transcribe(
                temp_audio_path,
                word_timestamps=True,
                verbose=False
            )

            self.full_transcription = result["text"].strip()
            self.transcription_segments = result["segments"]

            # Save transcription to file
            transcription_data = {
                "video_file": str(self.video_path),
                "transcription_date": datetime.now().isoformat(),
                "model": "small",
                "full_transcription": self.full_transcription,
                "segments": self.transcription_segments,
                "duration": result.get("duration", 0)
            }

            try:
                with open(transcription_path, 'w') as f:
                    json.dump(transcription_data, f, indent=2)
                self.logger.info(f"💾 Transcription saved to: {transcription_path}")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not save transcription: {e}")

            # Log the transcription
            self.llm_responses.append({
                "timestamp": datetime.now().isoformat(),
                "type": "full_video_transcription",
                "audio_file": temp_audio_path,
                "full_transcription": self.full_transcription,
                "num_segments": len(self.transcription_segments),
                "duration": result.get("duration", 0),
                "saved_to": str(transcription_path)
            })

            # Clean up temp file
            os.unlink(temp_audio_path)

            transcription_time = time.time() - transcription_start
            minutes = int(transcription_time // 60)
            seconds = int(transcription_time % 60)

            self.logger.info(f"✅ Full video transcription complete!")
            self.logger.info(f"   📊 Found {len(self.transcription_segments)} speech segments")
            self.logger.info(f"   ⏱️  Transcription took: {minutes}m {seconds}s")
            self.logger.info(f"   📝 Total transcript length: {len(self.full_transcription)} characters")
            self.logger.info(f"{'='*60}")
            return True

        except Exception as e:
            self.logger.error(f"Error transcribing full video: {e}")
            return False

    def get_transcription_for_timerange(self, start_time: float, end_time: float) -> Optional[str]:
        """Get transcribed text for a specific time range from cached full transcription."""
        if not self.transcription_segments:
            return None

        relevant_text = []

        for segment in self.transcription_segments:
            seg_start = segment["start"]
            seg_end = segment["end"]

            # Check if segment overlaps with our time range
            if seg_end >= start_time and seg_start <= end_time:
                relevant_text.append(segment["text"].strip())

        combined_text = " ".join(relevant_text).strip()

        if combined_text:
            # Log the segment extraction
            self.llm_responses.append({
                "timestamp": datetime.now().isoformat(),
                "type": "transcription_segment_extraction",
                "time_range": f"{start_time:.2f}s - {end_time:.2f}s",
                "extracted_text": combined_text
            })

        return combined_text if combined_text else None

    def call_lm_studio_with_images(self, prompt: str, system_message: str, images_base64: List[str]) -> Optional[str]:
        """Make a request to LM Studio API with image data."""
        # First try with vision support
        try:
            # Prepare content with text and images
            content = [{"type": "text", "text": prompt}]
            for img_data in images_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_data}"
                    }
                })

            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_message} if system_message else {},
                    {"role": "user", "content": content}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }

            # Remove empty system message
            payload["messages"] = [msg for msg in payload["messages"] if msg.get("content")]

            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                content_result = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Log the LLM response (without image data for size reasons)
                self.llm_responses.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "vision",
                    "prompt": prompt,
                    "system_message": system_message,
                    "num_images": len(images_base64),
                    "image_sizes": [len(img) for img in images_base64],
                    "response": content_result
                })

                return content_result
            else:
                self.logger.warning(f"LM Studio vision API error ({response.status_code}), falling back to text-only")
                # Fallback to text-only if vision not supported
                return self._fallback_image_to_text(prompt, system_message, len(images_base64))

        except Exception as e:
            self.logger.warning(f"Error calling LM Studio with images ({e}), falling back to text-only")
            # Fallback to text-only
            return self._fallback_image_to_text(prompt, system_message, len(images_base64))

    def _fallback_image_to_text(self, prompt: str, system_message: str, num_images: int) -> Optional[str]:
        """Fallback when vision is not supported."""
        fallback_prompt = f"{prompt}\n\n[Note: {num_images} video frames were extracted but cannot be processed by current LM Studio model. Please provide text-based analysis assuming typical NFL broadcast graphics.]"

        # Log the fallback
        self.llm_responses.append({
            "timestamp": datetime.now().isoformat(),
            "type": "vision_fallback",
            "prompt": prompt,
            "system_message": system_message,
            "num_images": num_images,
            "fallback_reason": "LM Studio doesn't support vision input",
            "response": "Fell back to text-only processing"
        })

        return self.call_lm_studio(fallback_prompt, system_message)


    def extract_audio_frame(self, start_time: float, end_time: float) -> Optional[str]:
        """Get transcribed audio for a video segment from cached full transcription."""
        self.logger.info(f"Extracting transcription from {start_time:.2f}s to {end_time:.2f}s")
        return self.get_transcription_for_timerange(start_time, end_time)

    def extract_video_frames(self, start_time: float, end_time: float, num_frames: int = 3) -> List[str]:
        """Extract video frames from a segment and encode as base64."""
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame positions
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            frame_step = max(1, (end_frame - start_frame) // num_frames)

            frames_base64 = []

            for i in range(num_frames):
                frame_pos = start_frame + (i * frame_step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)

                    # Save to temporary file and encode
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                        pil_image.save(temp_img.name, "JPEG", quality=85)

                        with open(temp_img.name, "rb") as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            frames_base64.append(img_base64)

                        # Clean up temp file
                        os.unlink(temp_img.name)

            cap.release()
            return frames_base64

        except Exception as e:
            self.logger.error(f"Error extracting video frames: {e}")
            return []

    def verify_football_play(self, start_time: float, end_time: float) -> bool:
        """Ask LLM to verify if this clip shows an actual football play."""
        system_msg = "You are an expert at analyzing NFL game footage. Determine if the provided video frames show an actual football play in progress."

        # Extract video frames for verification
        frames_base64 = self.extract_video_frames(start_time, end_time)
        if not frames_base64:
            return False

        prompt = f"""
        Please analyze these video frames from an NFL broadcast (from {start_time:.2f}s to {end_time:.2f}s).

        Determine if this clip shows an ACTUAL FOOTBALL PLAY in progress. This means:
        - Players lined up and executing a play (running, passing, kicking)
        - Active game action, not commercials, replays, or commentary
        - Field view showing the play development

        DO NOT consider these as football plays:
        - Commercials or advertisements
        - Studio commentary or analysis
        - Crowd shots or sideline shots
        - Replay analysis or slow-motion breakdowns
        - Pre-snap or post-play celebrations only

        Respond with only "YES" if this shows an actual football play, or "NO" if it does not.
        """

        result = self.call_lm_studio_with_images(prompt, system_msg, frames_base64)
        is_play = result and result.strip().upper() == "YES"

        # Log the verification
        self.llm_responses.append({
            "timestamp": datetime.now().isoformat(),
            "type": "football_play_verification",
            "time_range": f"{start_time:.2f}s - {end_time:.2f}s",
            "is_football_play": is_play,
            "llm_response": result
        })

        return is_play

    def extract_screen_text(self, start_time: float, end_time: float) -> Optional[str]:
        """Extract text displayed on screen from a video segment."""
        system_msg = "You are an expert at reading text from NFL game broadcasts. Extract all visible text including scores, player names, statistics, down and distance, time remaining, and any other on-screen information."

        # Extract actual video frames
        frames_base64 = self.extract_video_frames(start_time, end_time)
        if not frames_base64:
            return None

        prompt = f"""
        Please identify and extract all text visible on screen in these NFL video frames (from {start_time:.2f}s to {end_time:.2f}s).
        Include:
        - Score displays
        - Down and distance
        - Time remaining
        - Player names and numbers
        - Team names
        - Statistics
        - Any other on-screen text or graphics

        Return only the extracted text, no additional commentary.
        """

        return self.call_lm_studio_with_images(prompt, system_msg, frames_base64)

    def parse_screen_data(self, screen_text: str) -> Dict:
        """Parse structured data from screen text."""
        data = {
            'quarter': None,
            'down': None,
            'distance': None,
            'time': None,
            'score_home': None,
            'score_away': None
        }

        if not screen_text:
            return data

        text_lower = screen_text.lower()

        # Parse quarter
        import re
        quarter_match = re.search(r'(?:quarter:|qtr:?|^)\s*(1st|2nd|3rd|4th|ot|overtime)', text_lower)
        if quarter_match:
            q = quarter_match.group(1)
            data['quarter'] = {'1st': '1', '2nd': '2', '3rd': '3', '4th': '4', 'ot': '5', 'overtime': '5'}.get(q)

        # Parse down and distance
        down_dist_match = re.search(r'(1st|2nd|3rd|4th)\s*&\s*(\d+)', text_lower)
        if down_dist_match:
            data['down'] = {'1st': '1', '2nd': '2', '3rd': '3', '4th': '4'}.get(down_dist_match.group(1))
            data['distance'] = down_dist_match.group(2)

        # Parse time (various formats)
        time_match = re.search(r'time[^\d]*([0-9]{1,2}):([0-9]{2})', text_lower)
        if not time_match:
            time_match = re.search(r'([0-9]{1,2}):([0-9]{2})', screen_text)  # Case sensitive for time
        if time_match:
            data['time'] = f"{time_match.group(1)}:{time_match.group(2)}"

        return data

    def extract_player_names(self, audio_text: str) -> List[str]:
        """Extract potential player names from audio transcription."""
        if not audio_text:
            return []

        # Common NFL player name patterns (last names)
        import re

        # Split into words and look for capitalized names
        words = audio_text.split()
        potential_names = []

        for word in words:
            # Look for capitalized words that could be names (2+ chars, not common words)
            if (len(word) >= 2 and word[0].isupper() and
                word.lower() not in ['the', 'and', 'for', 'down', 'yard', 'yards', 'touchdown', 'first', 'second', 'third', 'fourth']):
                potential_names.append(word.strip('.,!?'))

        return potential_names

    def match_play_data(self, audio_text: str, screen_text: str, scene_start: float, scene_end: float) -> int:
        """Match extracted text against game log play data and track unique plays."""
        new_matches = 0

        if not audio_text and not screen_text:
            return new_matches

        # Verify this is actually a football play before attempting matches
        if not self.verify_football_play(scene_start, scene_end):
            self.logger.info("     ❌ Not a football play - skipping matching")
            return new_matches

        combined_text = f"{audio_text or ''} {screen_text or ''}".lower()

        for play_index, play in enumerate(self.game_log):
            play_matches = []

            # Check various fields for matches
            fields_to_check = ['desc', 'time', 'yrdln']

            for field in fields_to_check:
                field_value = str(play.get(field, '')).lower()
                if field_value and len(field_value) > 2 and field_value in combined_text:
                    play_matches.append({
                        'field': field,
                        'value': play.get(field),
                        'confidence': len(field_value) / len(combined_text)
                    })

            # Check player name matches
            if audio_text:
                extracted_names = self.extract_player_names(audio_text)
                player_fields = [field for field in play.keys() if field.endswith('_player_name')]

                for player_field in player_fields:
                    player_name = play.get(player_field, '')
                    if player_name:
                        # Check if any extracted name matches the last name
                        player_last_name = player_name.split()[-1] if player_name else ''
                        for extracted_name in extracted_names:
                            if (len(player_last_name) > 2 and
                                extracted_name.lower() == player_last_name.lower()):
                                play_matches.append({
                                    'field': player_field,
                                    'value': player_name,
                                    'confidence': 0.3  # High confidence for name matches
                                })
                                break

            if play_matches:
                total_confidence = sum(match['confidence'] for match in play_matches)
                play_id = play.get('play_id')

                # Check if we already have a match for this play
                if play_id in self.play_matches:
                    # Compare confidence scores
                    if total_confidence > self.play_matches[play_id]['total_confidence']:
                        # Replace with higher confidence match
                        self.play_matches[play_id] = self._format_match_result(
                            play_index, scene_start, scene_end, play, play_matches,
                            audio_text, screen_text, total_confidence
                        )
                        self.logger.info(f"     🔄 Updated Play {play_id} with higher confidence: {total_confidence:.3f}")
                else:
                    # New play match
                    self.play_matches[play_id] = self._format_match_result(
                        play_index, scene_start, scene_end, play, play_matches,
                        audio_text, screen_text, total_confidence
                    )
                    new_matches += 1

        return new_matches

    def _format_match_result(self, play_index: int, scene_start: float, scene_end: float,
                           play: Dict, matches: List[Dict], audio_text: str,
                           screen_text: str, total_confidence: float) -> Dict:
        """Format a match result according to the specified structure."""
        # Extract key player names and info
        play_data = {
            "play_id": play.get('play_id'),
            "game_id": play.get('game_id'),
            "home_team": play.get('home_team'),
            "away_team": play.get('away_team'),
            "week": play.get('week'),
            "posteam": play.get('posteam'),
            "defteam": play.get('defteam'),
            "qtr": play.get('qtr'),
            "down": play.get('down'),
            "time": play.get('time'),
            "yrdln": play.get('yrdln'),
            "ydstogo": play.get('ydstogo'),
            "play_type": play.get('play_type'),
            "yards_gained": play.get('yards_gained'),
            "passer_player_name": play.get('passer_player_name', ''),
            "receiver_player_name": play.get('receiver_player_name', ''),
            "rusher_player_name": play.get('rusher_player_name', ''),
            "interception_player_name": play.get('interception_player_name', ''),
            "punt_returner_player_name": play.get('punt_returner_player_name', ''),
            "kickoff_returner_player_name": play.get('kickoff_returner_player_name', ''),
            "punter_player_name": play.get('punter_player_name', ''),
            "kicker_player_name": play.get('kicker_player_name', ''),
            "blocked_player_name": play.get('blocked_player_name', ''),
            "tackle_for_loss_1_player_name": play.get('tackle_for_loss_1_player_name', ''),
            "tackle_for_loss_2_player_name": play.get('tackle_for_loss_2_player_name', ''),
            "qb_hit_1_player_name": play.get('qb_hit_1_player_name', ''),
            "qb_hit_2_player_name": play.get('qb_hit_2_player_name', ''),
            "forced_fumble_player_1_player_name": play.get('forced_fumble_player_1_player_name', ''),
            "forced_fumble_player_2_player_name": play.get('forced_fumble_player_2_player_name', ''),
            "solo_tackle_1_player_name": play.get('solo_tackle_1_player_name', ''),
            "solo_tackle_2_player_name": play.get('solo_tackle_2_player_name', ''),
            "assist_tackle_1_player_name": play.get('assist_tackle_1_player_name', ''),
            "assist_tackle_2_player_name": play.get('assist_tackle_2_player_name', ''),
            "assist_tackle_3_player_name": play.get('assist_tackle_3_player_name', ''),
            "assist_tackle_4_player_name": play.get('assist_tackle_4_player_name', ''),
            "tackle_with_assist_1_player_name": play.get('tackle_with_assist_1_player_name', ''),
            "tackle_with_assist_2_player_name": play.get('tackle_with_assist_2_player_name', ''),
            "pass_defense_1_player_name": play.get('pass_defense_1_player_name', ''),
            "pass_defense_2_player_name": play.get('pass_defense_2_player_name', ''),
            "fumbled_1_player_name": play.get('fumbled_1_player_name', ''),
            "fumbled_2_player_name": play.get('fumbled_2_player_name', ''),
            "fumble_recovery_1_player_name": play.get('fumble_recovery_1_player_name', ''),
            "fumble_recovery_2_player_name": play.get('fumble_recovery_2_player_name', ''),
            "sack_player_name": play.get('sack_player_name', ''),
            "half_sack_1_player_name": play.get('half_sack_1_player_name', ''),
            "half_sack_2_player_name": play.get('half_sack_2_player_name', ''),
            "penalty_player_name": play.get('penalty_player_name', ''),
            "safety_player_name": play.get('safety_player_name', ''),
            "season": play.get('season'),
            "play_clock": play.get('play_clock'),
            "play_type_nfl": play.get('play_type_nfl'),
        }

        return {
            "playIndex": play_index,
            "startTime": scene_start,
            "endTime": scene_end,
            "playDescription": play.get('desc', ''),
            "play": play_data,
            "matches": matches,
            "audio_text": audio_text or '',
            "screen_text": screen_text or '',
            "total_confidence": total_confidence
        }

    def process_video(self):
        """Main processing function."""
        self.logger.info(f"Processing video: {self.video_path}")
        self.logger.info(f"Using game log: {self.game_log_path}")

        # Transcribe the entire video first (much more efficient)
        if not self.transcribe_full_video():
            self.logger.error("Failed to transcribe video. Exiting.")
            return

        # Detect scene changes
        self.logger.info(f"\n🎬 Detecting scene changes...")
        scenes = self.detect_scene_changes()
        if not scenes:
            self.logger.error("❌ No scenes detected. Exiting.")
            return

        self.logger.info(f"✅ Found {len(scenes)} scenes to process")
        total_duration = scenes[-1][1] - scenes[0][0] if scenes else 0
        self.logger.info(f"📏 Video duration to analyze: {total_duration:.1f} seconds")

        # If debug mode, save scenes and exit
        if self.debug_scenes:
            self.logger.info(f"\n🐛 DEBUG MODE: Saving detected scenes with audio transcription and exiting...")
            self.save_debug_scenes_with_audio(scenes)
            scenes_over_10s = sum(1 for start, end in scenes if (end - start) >= 10.0)
            self.logger.info(f"\n📊 SCENE ANALYSIS COMPLETE:")
            self.logger.info(f"   Total scenes: {len(scenes)}")
            self.logger.info(f"   Scenes ≥10s: {scenes_over_10s}")
            self.logger.info(f"   Scenes <10s: {len(scenes) - scenes_over_10s}")
            self.logger.info(f"   Total duration: {total_duration:.1f}s")
            self.logger.info(f"🐛 Debug file created in processing_results/ folder")
            return
        # Process each scene with progress tracking
        total_scenes = len(scenes)
        start_processing_time = time.time()
        scene_times = []
        total_matches = 0

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING SCENE PROCESSING - {total_scenes} scenes to analyze")
        self.logger.info(f"{'='*60}")

        for i, (start_time, end_time) in enumerate(scenes):
            scene_start_time = time.time()
            scene_num = i + 1

            # Calculate progress and ETA
            progress_pct = (i / total_scenes) * 100
            if i > 0:
                avg_time_per_scene = sum(scene_times) / len(scene_times)
                remaining_scenes = total_scenes - scene_num
                eta_seconds = remaining_scenes * avg_time_per_scene
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f" | ETA: {eta_min}m {eta_sec}s"
            else:
                eta_str = " | ETA: calculating..."

            self.logger.info(f"\n[{progress_pct:5.1f}%] Scene {scene_num}/{total_scenes}: {start_time:.1f}s-{end_time:.1f}s{eta_str}")

            # Skip scenes shorter than 15 seconds
            scene_duration = end_time - start_time
            if scene_duration < 15.0:
                self.logger.info(f" Scene too short ({scene_duration:.1f}s) - skipping")
                # Track timing for ETA but don't process further
                scene_total_time = time.time() - scene_start_time
                scene_times.append(scene_total_time)
                if len(scene_times) > 15:
                    scene_times = scene_times[-15:]
                continue

            # Extract audio transcription
            audio_start = time.time()
            audio_text = self.extract_audio_frame(start_time, end_time)
            audio_time = time.time() - audio_start

            if audio_text:
                audio_preview = audio_text[:50] + ('...' if len(audio_text) > 50 else '')
                self.logger.info(f"  ✅ Audio transcribed ({audio_time:.1f}s): \"{audio_preview}\"")
            else:
                self.logger.info(f" No audio transcription found ({audio_time:.1f}s)")

            # Extract screen text
            screen_start = time.time()
            screen_text = self.extract_screen_text(start_time, end_time)
            screen_time = time.time() - screen_start

            if screen_text:
                screen_preview = screen_text[:50] + ('...' if len(screen_text) > 50 else '')
                self.logger.info(f"  ✅ Screen text extracted ({screen_time:.1f}s): \"{screen_preview}\"")
            else:
                self.logger.info(f" No screen text found ({screen_time:.1f}s)")

            # Match against play data
            match_start = time.time()
            new_matches = self.match_play_data(audio_text, screen_text, start_time, end_time)
            match_time = time.time() - match_start

            if new_matches > 0:
                total_matches = len(self.play_matches)
                self.logger.info(f"  🎯 Found {new_matches} new matches! ({match_time:.1f}s) [Total unique: {total_matches}]")
            else:
                self.logger.info(f"  ❌ No new matches found ({match_time:.1f}s)")

            # Track timing for ETA calculation
            scene_total_time = time.time() - scene_start_time
            scene_times.append(scene_total_time)

            # Keep only recent times for better ETA accuracy
            if len(scene_times) > 10:
                scene_times = scene_times[-10:]

            self.logger.info(f"  ⏱️  Scene completed in {scene_total_time:.1f}s")

        # Processing summary
        total_processing_time = time.time() - start_processing_time
        minutes = int(total_processing_time // 60)
        seconds = int(total_processing_time % 60)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PROCESSING COMPLETE!")
        self.logger.info(f"{'='*60}")
        # Convert play_matches to results list for output
        self.results = list(self.play_matches.values())

        self.logger.info(f"📊 Total scenes processed: {total_scenes}")
        self.logger.info(f"🎯 Total unique plays matched: {len(self.results)}")
        self.logger.info(f"⏱️  Total processing time: {minutes}m {seconds}s")
        self.logger.info(f"⚡ Average time per scene: {total_processing_time/total_scenes:.1f}s")

        if self.results:
            # Show top matches
            top_matches = sorted(self.results, key=lambda m: m['total_confidence'], reverse=True)[:3]
            self.logger.info(f"\n🏆 TOP MATCHES:")
            for i, match in enumerate(top_matches, 1):
                self.logger.info(f"  {i}. Time {match['startTime']:.1f}s - Play {match['play']['play_id']} - {match['playDescription'][:50]}...")

        # Save results
        self.save_results()
        self.save_llm_responses()

        self.logger.info(f"\n💾 Results saved to processing_results/ folder")
        self.logger.info(f"{'='*60}")

    def save_results(self):
        """Save processing results to JSON file in the specified format."""
        output_file = Path("processing_results") / f"results_{self.run_id}.json"

        # Extract game information from the first play if available
        game_info = "Unknown Game"
        video_url = str(self.video_path)  # Local path to the video file

        if self.game_log and len(self.game_log) > 0:
            first_play = self.game_log[0]
            home_team = first_play.get('home_team', '')
            away_team = first_play.get('away_team', '')
            week = first_play.get('week', '')
            season = first_play.get('season', '')

            if week and season:
                if week in ['18', '19', '20', '21']:  # Playoff weeks
                    week_name = {'18': 'Wild Card', '19': 'Divisional', '20': 'Conference Championship', '21': 'Super Bowl'}.get(week, f"Week {week}")
                else:
                    week_name = f"Week {week}"
                game_info = f"{season} {week_name} - {away_team} @ {home_team}"
            else:
                game_info = f"{away_team} @ {home_team}"

        result_data = {
            "game": game_info,
            "videoUrl": video_url,
            "clips": sorted(self.results, key=lambda x: x['startTime'])  # Sort by start time
        }

        try:
            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            self.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def save_llm_responses(self):
        """Save all LLM responses to a log file."""
        log_file = Path("processing_results") / f"llm_responses_{self.run_id}.json"

        try:
            with open(log_file, 'w') as f:
                json.dump(self.llm_responses, f, indent=2)
            self.logger.info(f"LLM responses saved to {log_file}")
        except Exception as e:
            self.logger.error(f"Error saving LLM responses: {e}")

    def save_debug_scenes_with_audio(self, scenes: List[Tuple[float, float]]):
        """Save detected scenes with audio transcription to JSON file for debugging."""
        output_file = Path("processing_results") / f"debug_scenes_{self.run_id}.json"

        # Extract game information from the first play if available
        game_info = "Unknown Game"
        if self.game_log and len(self.game_log) > 0:
            first_play = self.game_log[0]
            home_team = first_play.get('home_team', '')
            away_team = first_play.get('away_team', '')
            week = first_play.get('week', '')
            season = first_play.get('season', '')

            if week and season:
                if week in ['18', '19', '20', '21']:  # Playoff weeks
                    week_name = {'18': 'Wild Card', '19': 'Divisional', '20': 'Conference Championship', '21': 'Super Bowl'}.get(week, f"Week {week}")
                else:
                    week_name = f"Week {week}"
                game_info = f"{season} {week_name} - {away_team} @ {home_team}"
            else:
                game_info = f"{away_team} @ {home_team}"

        # Process scenes with audio transcription
        scenes_data = []
        scenes_over_10s = 0

        for i, (start_time, end_time) in enumerate(scenes):
            duration = end_time - start_time
            is_long_enough = duration >= 10.0

            if is_long_enough:
                scenes_over_10s += 1

            # Get audio transcription for this scene
            self.logger.info(f"   Processing scene {i+1}/{len(scenes)}: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")

            audio_text = None
            if is_long_enough:  # Only transcribe scenes that meet minimum duration
                audio_text = self.get_transcription_for_timerange(start_time, end_time)
                if audio_text:
                    preview = audio_text[:100] + ('...' if len(audio_text) > 100 else '')
                    self.logger.info(f"     ✅ Audio: \"{preview}\"")
                else:
                    self.logger.info(f"    No audio found")
            else:
                self.logger.info(f"    Scene too short - skipping transcription")

            # Skip short scenes if --no-short flag is enabled
            if self.no_short and not is_long_enough:
                continue

            scenes_data.append({
                "sceneIndex": i,
                "startTime": start_time,
                "endTime": end_time,
                "duration": duration,
                "isLongEnough": is_long_enough,
                "audioTranscript": audio_text or ""
            })

        debug_data = {
            "game": game_info,
            "videoUrl": str(self.video_path),
            "totalScenes": len(scenes),
            "scenesOver10s": scenes_over_10s,
            "totalDuration": scenes[-1][1] - scenes[0][0] if scenes else 0,
            "scenes": scenes_data
        }

        try:
            with open(output_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            self.logger.info(f"🐛 Debug scenes with audio saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving debug scenes: {e}")

    def save_debug_scenes(self, scenes: List[Tuple[float, float]]):
        """Save detected scenes to JSON file for debugging."""
        output_file = Path("processing_results") / f"debug_scenes_{self.run_id}.json"

        # Extract game information from the first play if available
        game_info = "Unknown Game"
        if self.game_log and len(self.game_log) > 0:
            first_play = self.game_log[0]
            home_team = first_play.get('home_team', '')
            away_team = first_play.get('away_team', '')
            week = first_play.get('week', '')
            season = first_play.get('season', '')

            if week and season:
                if week in ['18', '19', '20', '21']:  # Playoff weeks
                    week_name = {'18': 'Wild Card', '19': 'Divisional', '20': 'Conference Championship', '21': 'Super Bowl'}.get(week, f"Week {week}")
                else:
                    week_name = f"Week {week}"
                game_info = f"{season} {week_name} - {away_team} @ {home_team}"
            else:
                game_info = f"{away_team} @ {home_team}"

        # Create scenes list
        scenes_data = []
        for i, (start_time, end_time) in enumerate(scenes):
            duration = end_time - start_time
            scenes_data.append({
                "sceneIndex": i,
                "startTime": start_time,
                "endTime": end_time,
                "duration": duration,
                "isLongEnough": duration >= 10.0  # Flag for 10-second minimum
            })

        debug_data = {
            "game": game_info,
            "videoUrl": str(self.video_path),
            "totalScenes": len(scenes),
            "scenesOver10s": sum(1 for scene in scenes_data if scene["isLongEnough"]),
            "totalDuration": scenes[-1][1] - scenes[0][0] if scenes else 0,
            "scenes": scenes_data
        }

        try:
            with open(output_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            self.logger.info(f"🐛 Debug scenes saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving debug scenes: {e}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Process NFL videos to detect timestamps")
    parser.add_argument("video_file", help="Path to the video file")
    parser.add_argument("game_log", help="Path to the game log JSON file")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234",
                       help="LM Studio API URL (default: http://localhost:1234)")
    parser.add_argument("--debug-scenes", action="store_true",
                       help="Output detected scenes to JSON for debugging (skips LLM processing)")
    parser.add_argument("--no-short", action="store_true",
                       help="When used with --debug-scenes, only include scenes ≥10 seconds")
    parser.add_argument("--transcription", type=str,
                       help="Path to existing transcription JSON file (skips re-transcribing)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.video_file).exists():
        print(f"Error: Video file not found: {args.video_file}")
        sys.exit(1)

    if not Path(args.game_log).exists():
        print(f"Error: Game log file not found: {args.game_log}")
        sys.exit(1)

    # Create processor and run
    processor = NFLProcessor(args.video_file, args.game_log, args.lm_studio_url, args.debug_scenes, args.no_short, args.transcription)
    processor.process_video()


if __name__ == "__main__":
    main()