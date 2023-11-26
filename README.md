# Tanka Poetry Generator

## Description

The Tanka Poetry Generator is an innovative system that uses computer vision and natural language processing to generate Tanka poems from user-uploaded videos. It analyzes video content to identify key scenes and objects and then creatively constructs a Tanka poem reflecting the video's essence. 

## Project Structure

### HTML, CSS, and JavaScript
- **HTML Pages:**
  - `index.html`: Main page for uploading videos and viewing previously generated Tanka poems.
  - `display_video.html`: Displays the uploaded video alongside the generated Tanka poem.
  - `previously_generated_tanka.html`: Shows selected poems from past uploads.
- **CSS Files:**
  - `index.css` and `display_video.css`: Define styles for the respective HTML pages.
- **JavaScript Files:**
  - `index.js`: Manages uploads and interactions on `index.html`.
  - `display_video.js`: Controls display of poems and videos on `display_video.html`.

### Python (`website.py`)
- Backend logic, including video processing, scene and object identification, Tanka poem generation, and web server functionalities.
- Utilizes Flask, OpenCV, PyTorch, and NLTK.

### Flow and Linking of Web Pages
- `index.html` for uploading videos or viewing past poems.
- `website.py` processes uploaded videos and displays the poem on `display_video.html`.
- Navigate back to `index.html` or view a previously generated poem on `previously_generated_tanka.html`.

## Setup and Running the Code

1. **Environment Setup:**
   - Install Python and libraries: Flask, OpenCV, PyTorch, NLTK.
   - Set up a virtual environment.

2. **Running the Application:**
   - Run `python website.py` to start the Flask server.
   - Access the web application at `http://127.0.0.1:5000/`.

## Challenges Faced

- Integrating computer vision with natural language processing.
- Adhering to traditional Tanka syllable structure while reflecting video content.
- Balancing technical accuracy with poetic creativity.

## Scholarly Papers That Inspired This Project

1. "Computer Vision in Robotics and Industrial Applications" - Insights into computer vision for video analysis.
2. "Natural Language Processing and Its Uses in Computational Linguistics" - Understanding NLP for linguistic construction of Tanka poems.
3. "Creative Systems: A Computational Approach to Understanding Creativity in Art" - Blending technology and creativity for poetry generation.

These papers inspired the approach of combining computer vision and NLP to create a system that analyzes visual content and translates it into creative poetic form.
