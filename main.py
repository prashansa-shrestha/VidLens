from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from moviepy import VideoFileClip
import whisper
import io
import os
import cv2
import numpy as np
from typing import Tuple, List
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import tempfile

# Initialize FastAPI app first
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remove any existing log handlers
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    logger.removeHandler(handler)

# Configure console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configure file logging
file_handler = RotatingFileHandler('app.log', maxBytes=10 * 1024 * 1024, backupCount=10)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Set the log level for the application
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env
load_dotenv()

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Get API key from environment variable
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable is not set")

# Configure the API key
genai.configure(api_key=api_key)

class VideoClipAnalyzer:
    def __init__(self, threshold: float = 30.0, min_clip_length: int = 1):
        self.threshold = threshold
        self.min_clip_length = min_clip_length
        self.logger = logger

    def process_video(self, video_path: str) -> List[np.ndarray]:
        """Process video file and return frames from each clip"""
        self.logger.info(f"Processing video: {video_path}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Error: Could not open video file")
            return []

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video FPS: {fps}")
        self.logger.info(f"Total frames: {total_frames}")

        # Initialize variables
        prev_frame = None
        clip_boundaries = []
        frame_count = 0
        current_clip_length = 0
        clip_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate frame difference
                frame_diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(frame_diff)

                # Detect scene change
                if mean_diff > self.threshold and current_clip_length >= self.min_clip_length:
                    clip_boundaries.append((frame_count - current_clip_length, frame_count))
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip_frames.append(frame_rgb)
                    current_clip_length = 0
                
            current_clip_length += 1
            prev_frame = gray.copy()
            frame_count += 1

        # Add the last clip
        if current_clip_length >= self.min_clip_length:
            clip_boundaries.append((frame_count - current_clip_length, frame_count))
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clip_frames.append(frame_rgb)

        cap.release()
        self.logger.info(f"Found {len(clip_frames)} clips")
        
        return clip_frames

def format_storytelling_analysis(analysis_text: str) -> str:
    """Format the storytelling analysis into readable sections."""
    sections = analysis_text.split('\n\n')
    formatted_text = ""
    
    # Format the analysis section
    if len(sections) >= 1:
        formatted_text += "📝 Analysis of your story:\n\n"
        formatted_text += sections[0].strip() + "\n\n"
    
    # Format the improvement suggestions
    if len(sections) >= 2:
        formatted_text += "✨ Suggestions for improvement:\n\n"
        formatted_text += sections[1].strip() + "\n\n"
    
    # Format the alternative script
    if len(sections) >= 3:
        formatted_text += "📚 Suggested alternative script:\n\n"
        formatted_text += sections[2].strip()
    
    return formatted_text

def format_mood_analysis(analysis_text: str, clip_number: int = None) -> str:
    """Format the mood analysis into a readable format."""
    if clip_number is not None:
        header = f"🎬 Scene {clip_number}:\n\n"
    else:
        header = "🎨 Image Analysis:\n\n"
    
    # Split analysis into points and format them
    points = analysis_text.split('\n')
    formatted_points = "\n".join(f"• {point.strip()}" for point in points if point.strip())
    
    return f"{header}{formatted_points}\n"

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video file."""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        video.close()
        audio.close()
    except Exception as e:
        logger.error(f"Error extracting audio: {e}", exc_info=True)
        raise

def analyze_storytelling(transcript: str):
    """Analyze the storytelling method using Gemini."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # First analyze the current storytelling
        analysis_prompt = f"""Analyze the storytelling method in this transcript:
        {transcript}
        
        Please evaluate:
        1. Narrative structure
        2. Character development
        3. Pacing and flow
        4. Key themes and messages
        5. Engagement techniques used

        Provide your analysis in clear, concise points.
        """
        
        analysis_response = model.generate_content(analysis_prompt)
        
        # Then generate an improved script
        script_prompt = f"""Based on this transcript:
        {transcript}
        
        Create an improved version that maintains the same core message but enhances:
        1. Narrative structure
        2. Character development
        3. Pacing and flow
        4. Engagement
        5. Emotional impact
        
        Write the new script in a natural, conversational style.
        Keep it approximately the same length as the original.
        """
        
        script_response = model.generate_content(script_prompt)
        
        return {
            "analysis": analysis_response.text,
            "alternative_script": script_response.text
        }
    except Exception as e:
        logger.error(f"Error analyzing storytelling: {e}", exc_info=True)
        raise

def format_storytelling_analysis(analysis_dict: dict) -> str:
    """Format the storytelling analysis into readable sections."""
    formatted_text = ""
    
    # Format the analysis section
    formatted_text += "📝 Analysis of your story:\n\n"
    formatted_text += analysis_dict["analysis"].strip() + "\n\n"
    
    # Format the alternative script
    formatted_text += "📚 Suggested alternative script:\n\n"
    formatted_text += analysis_dict["alternative_script"].strip()
    
    return formatted_text
def analyze_mood(image_data, mood: str = "horror"):
    """Analyze the mood of an image using Gemini."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Analyze if this image is in a {mood} vibe. 
            If it is good and no improvements are needed for color grading, say so. 
            If the image is not in the {mood} vibe, provide specific instructions for adjusting photo editing parameters. 
            (e.g., level of brightness, exposure, contrast, saturation, etc.) to make it match the {mood} vibe. Keep the answers concise and in points
"""

        response = model.generate_content(
            [prompt, {"mime_type": "image/png", "data": image_data}]
        )
        return response.text
    except Exception as e:
        logger.error(f"Error analyzing mood: {e}", exc_info=True)
        raise

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...), mood: str = "horror"):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            video_path = os.path.join(temp_dir, "video.mp4")
            audio_path = os.path.join(temp_dir, "audio.mp3")
            
            with open(video_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Extract audio
            logger.info("Extracting audio from video...")
            extract_audio(video_path, audio_path)
            
            # Process video into clips
            logger.info("Processing video into clips...")
            analyzer = VideoClipAnalyzer()
            clip_frames = analyzer.process_video(video_path)
            
            # Analyze mood for each clip
            logger.info("Analyzing mood for each clip...")
            clip_analyses = []
            for i, frame in enumerate(clip_frames):
                # Convert numpy array to bytes
                frame_pil = Image.fromarray(frame)
                frame_byte_arr = io.BytesIO()
                frame_pil.save(frame_byte_arr, format='PNG')
                frame_byte_arr = frame_byte_arr.getvalue()
                
                # Analyze mood
                mood_analysis = analyze_mood(frame_byte_arr, mood)
                clip_analyses.append({
                    "clip_number": i + 1,
                    "mood_analysis": mood_analysis
                })
            
            # Transcribe audio
            logger.info("Transcribing audio...")
            result = whisper_model.transcribe(audio_path)
            transcript = result["text"]
            
            # Analyze storytelling
            logger.info("Analyzing storytelling...")
            storytelling_analysis = analyze_storytelling(transcript)
            
            # Format the response sections
            formatted_response = {
                "summary": {
                    "title": "📹 Video Analysis Summary",
                    "content": f"I've analyzed your video and found {len(clip_frames)} distinct scenes. "
                              f"Here's what I found:"
                },
                "transcript": {
                    "title": "🎤 Video Transcript",
                    "content": transcript
                },
                "storytelling": {
                    "title": "📖 Storytelling Analysis",
                    "content": format_storytelling_analysis(storytelling_analysis)
                },
                "scenes": {
                    "title": "🎭 Scene-by-Scene Mood Analysis",
                    "content": "\n".join(
                        format_mood_analysis(analysis["mood_analysis"], i + 1)
                        for i, analysis in enumerate(clip_analyses)
                    )
                }
            }
            
            return JSONResponse(content=formatted_response)
            
    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "title": "❌ Error",
                "message": "Sorry, there was a problem processing your video. Please try again."
            }
        )

@app.post("/mood-analyzer/")
async def analyze_image(file: UploadFile = File(...), mood: str = "horror"):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        analysis = analyze_mood(image_bytes, mood)
        formatted_response = {
            "title": "🖼️ Image Analysis Results",
            "content": format_mood_analysis(analysis)
        }
        
        return JSONResponse(content=formatted_response)
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "title": "❌ Error",
                "message": "Sorry, there was a problem analyzing your image. Please try again."
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)