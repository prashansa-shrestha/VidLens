# Video and Image Mood and Storytelling Analysis API
## [Demo Video](https://www.youtube.com/watch?v=BEu5NjoF79s).

This FastAPI project provides an API that analyzes videos and images in terms of mood and storytelling. It uses advanced AI models to transcribe audio, detect scene changes in videos, and evaluate the mood and structure of both images and video content.

## Features

- **Video Analysis**:

  - Extracts scenes from a video based on changes in frames.
  - Transcribes the video's audio into text.
  - Analyzes mood for each scene (e.g., horror, action, etc.).
  - Analyzes storytelling aspects such as narrative structure, pacing, and character development.

- **Image Mood Analysis**:
  - Analyzes the mood of an image and suggests adjustments (e.g., brightness, contrast) to match a specific mood (e.g., horror, calm).

## Requirements

- Python 3.7+
- Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Environment Variables

- **API_KEY**: Your API key for the generative AI model.

## API Endpoints

### 1. `/analyze-video/`

- **Method**: POST
- **Description**: Upload a video file to be analyzed.
- **Parameters**:
  - `file`: Video file (MP4).
  - `mood` (optional): The mood you want to analyze for (e.g., "horror").
- **Response**: Returns a JSON response with:
  - Video transcript.
  - Scene-by-scene mood analysis.
  - Storytelling analysis (narrative structure, pacing, character development).

### 2. `/mood-analyzer/`

- **Method**: POST
- **Description**: Upload an image to analyze its mood.
- **Parameters**:
  - `file`: Image file (PNG, JPEG, etc.).
  - `mood` (optional): The mood you want to analyze for (e.g., "horror").
- **Response**: Returns a JSON response with:
  - The analysis of the image's mood.
  - Suggestions for photo editing if the mood needs improvement.

## Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/project-name.git
   cd project-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   uvicorn main:app --reload
   ```

4. Open your browser and go to `http://127.0.0.1:8000` to access the API.

## Logging

Logs are written to both the console and a file (`app.log`). You can check the logs for detailed information on any issues during video processing, mood analysis, or storytelling evaluation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- FastAPI for the web framework.
- OpenCV for video processing.
- Whisper for audio transcription.
- Google Generative AI for mood and storytelling analysis.

```

This provides a simple overview of your API's capabilities, setup instructions, and usage examples. Adjust it according to your actual project details!
```
