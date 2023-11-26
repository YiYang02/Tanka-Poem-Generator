# Tanka Poetry Generator

## Description

The Tanka Poetry Generator is a computationally creative system that uses computer vision and natural language processing ideas to generate Tanka poems from user-uploaded videos. It first analyzes video content by splitting up the video into frames, then observes each frame for objects and key scenes via YOLOv3 and resnet152, respectively. Afterwards, it creates each Tanka line by using a pre-defined formula while picking words that fulfill the syllable count for that line. Lastly, it computes a score out of 5 that shows how well it was able to follow the Tanka poem format. 

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

## Setup and Running the Code

1. **Environment Setup:**
   - Install Python and libraries: Flask, OpenCV, PyTorch, NLTK.
   - run pip install -r requirements.txt within the project root directory
   - download https://pjreddie.com/media/files/yolov3.weights and place it within the /static folder. name it 'yolov3.weights'


2. **Running the Application:**
   - Run `python3 website.py` to start the Flask server.
   - Access the web application at `http://127.0.0.1:5000/`.
   - Have a video ready to upload adn enjoy your Tanka poem :)

## Challenges Faced and How I Grew as a Computer Scientist

There were a lot of challenges faced for me throughout the project from ideation to implementation and debugging. However, the 
general goal of this project was to enable me to grow as a computer scientist and push the bounds of my technical skills. I wanted
to create a form of poetry that wasn't so loosely defined that I couldn't just generated any words and be done. Rather, I wanted my poetry to have structure while still maintaining a high level of creativity. As such, I went with Tanka poems, where it is defined by 5 lines and the 5-7-5-7-7 syllable structure, but many of its themes revolves around nature, emotion, and self-reflection. 

Overall, my project was to create a computationally creative system that could take in video content (generally on nature and still videos) and create Tanka poems, which represents the natural poetry process of a Tanka poet from inspiration to planning to creation.
Consequently, I wanted the poem reader to experience emotional feelings of peace, tranquility and intra-reflection when reading the poem.

The first challenge was creating inspiration for my system, and so the task being able to parse a video and obtain key elements
and scenes. To do so, I had to research various computer vision libraries. It was hard finding libraries as some of their use cases
were too complex for my needs and others simply didn't do what I wanted. Moreso, some libraries were hard to use as their git repos were no longer being maintained. Overall, I went with YoloV3 and ResNet as these were population computer vision models and had plenty of resources online. I used YoloV3 for object detection and was able to detect objects within my video. For ResNet, I used 
MIT's hybrid1000 dataset, a labeled and categorized dataset of 1000 common scenes. It was also helpful that MIT had a Github up
with code of applying several models of ResNet via code I could reference. It was my first time doing such a comprehensive research of finding computer science libraries for a specific use case!

The second challenge was to then transition to the planning phase. I needed a way to generate a Tanka line by line, while adhereing to that line's syllable structure. Moreover, I wanted to follow a pre-defined formuala for generating a Tanka line. This is because I noticed that a lot of the Tanka nature poems generally use a combination of noun, verbs and adjectives to describe something within nature. Thus, I also had create datasets that mimicked parts of speech. The biggest challenge here was creating three datasets that had a variety of syllables for that part of speech (verb, adjective, adverb) which also ensuring that they are descriptive and "nature" related.

The third challenge was implementing my Tanka poem creation process! I had to now glue together all the helper functions, datasets I've added and created, and the libraries that I've imported and create a cohesive poem creation process. Here was where I spent the majority of my time debugging and adding/removing parts of the process. Through many iterations of my project I kept asking myself "How do I balance the technical accuracy with poetic creativity?" 

The fourth and final challenge was creating the website! This is my first time ever (besides Software Engineering class) creating on a full stack application from scratch. I had to ensure that the code I was writing was modular but also placed within a specific file that made structural sense. I also had to research and look up lots of HTML, CSS, Javascript documentation as my website has several workflows: from index.html to display_video.html and back, to index.html to previously_generated_tanka.hmtl. It was also hard for me to properly display the video and poem and so I had to learn and  around with CSS flexbox. Here, I grew the most as a full stack computer scientist!

## Scholarly Papers That Inspired This Project

1. Places: A 10 million Image Database for
Scene Recognition - http://places2.csail.mit.edu/PAMI_places.pdf 
This paper inspired me to realize that I can create a unique project and push myself by having a video input. Initially, I was going to do an image -> poem generator project, as I didn't know what sorts of computer vision work was out there and what comphrehensive datasets there were. This paper overcame my initial apphrension of creating a video input project.
2. As If Poetry: Computer-Generated Tanka and Contemporary Japanese Verse - https://www.asianetworkexchange.org/article/id/8145/
This paper inspired me to create a Tanka poem generator without relying on the state of the art tools and models. It also showed me that there is no video input capability which reinforced my project inspiration. 
3. YOLOv3: An Incremental Improvement - https://arxiv.org/abs/1804.02767
This paper inspired the object detection aspect of my poem creation process and how important it is to have a multi-model approach when creating computationally creative systems. The more resources a program can have at its disposable -- aka that of a human -- the more creative it possibly can be!
