"""
CSCI 3725 - Computational Creativity
M7: Poetry Jam
Author: Yi Yang
Date: 21 Nov, 2023
"""
import os
import cv2
import numpy as np
import shutil
import torch
import torch.nn.functional as F
import nltk
import random
import json

from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict

app = Flask(__name__)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('cmudict')
# nltk.download('punkt')

@app.route('/')
def index():
    saved_poems = load_poems()
    return render_template('index.html', poems=saved_poems)

@app.route('/view_poem')
def view_poem():
    poem_id = request.args.get('poem_id')
    poems = load_poems()
    selected_poem = next((poem for poem in poems if poem['id'] == poem_id), None)

    if selected_poem:
        return render_template('previously_generated_tanka.html', 
                               poem_data=selected_poem, 
                               video_filename=poem_id)
    else:
        return "Poem not found", 404

@app.route('/videos', methods=['POST'])
def upload_video():
   
    video_file = request.files['video']
    video_filename = secure_filename(video_file.filename)
    video_filepath = os.path.join('static', 'videos', video_filename)
    video_file.save(video_filepath)

    # Process video and extract frames. currently we save a frame every three seconds
    extract_frames(video_filepath, desired_seconds=3)

    # Identifies the key elements in the frames and returns them in a list
    objects_in_video = identify_objects_in_frames(os.path.join('static', 'frames'))

    scenes_in_video = identify_scene_in_frames(os.path.join('static', 'frames'))

    # generates the tanka poem with the objects and scenes from user video
    tanka_poem = create_tanka_poem(objects_in_video, scenes_in_video)
    
    # Save the poem with a unique identifier
    poem_data = {"id": video_filename, "poem": tanka_poem}
    save_poem(poem_data)

    print(evaluate_tanka_poem)

    return render_template('display_video.html', 
                        video_filename=video_filename,
                        first_line=tanka_poem[0],
                        second_line=tanka_poem[1],
                        third_line=tanka_poem[2],
                        fourth_line=tanka_poem[3],
                        fifth_line=tanka_poem[4])


"""
Takes the path to the uploaded video and extracts frames at a specified interval.
This allows for CV processing
"""
def extract_frames(video_path, desired_seconds=1):
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS) # Get video frame rate
    interval = int(frame_rate * desired_seconds)
    count = 0

    # we extract N intervals from the video thus getting N frames to process
    while success:
        if count % interval == 0:
            frame_filename = f"frame{count}.jpg"
            frame_filepath = os.path.join('static', 'frames', frame_filename)
            cv2.imwrite(frame_filepath, image)  # save frame as JPEG file
        success, image = video_cap.read()
        count += 1

"""
Identifies the major elements in the frames parsed from the video
"""
def identify_objects_in_frames(frames_folder):
    # Load YOLO configuration and weights
    yolo_net = cv2.dnn.readNet("static/yolov3.weights", "static/yolov3.cfg")
    layer_names = yolo_net.getLayerNames()

    # Get the indices of the output layers, then get the names
    out_layer_indices = yolo_net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in out_layer_indices.flatten()]

    # Load the COCO class labels
    with open("static/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    identified_objects = []

    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        height, width, _ = frame.shape

        # Convert the frame to a blob and perform a forward pass through YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)

        # Process YOLO outputs
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    label = str(classes[class_id])
                    identified_objects.append(label)
    
    return list(set(identified_objects))  # Removing duplicates

"""
Retrieves the top 3 likely scenes from each frame and return it. 
"""
def identify_scene_in_frames(frames_folder):
    # gets the index to category dict
    class_index = dict_index_to_class()

    # Load a pre-trained Places365 model (or an appropriate model)
    model = models.resnet152(pretrained=True)  # Replace with the correct model if necessary
    model.eval()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    identified_scenes = []

    # iterate through all the frames and identify objects
    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        image = Image.open(frame_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)
         # get the top 3 scenes in accordance to probabilities
        top3_prob, top3_catid = torch.topk(probabilities, 3)

        # gets the top three scenes for each frame
        for i in range(top3_prob.size(1)):
            class_id = top3_catid[0, i].item()
            if class_id in class_index:
                scene = class_index[class_id]
                prob = top3_prob[0, i].item()
                if prob > 0.1: # don't add scenes with low probability levels
                    scene_info = (scene, prob)
                    identified_scenes.append(scene_info)
                    # print(f"{i}: {scene} with probability {prob}")

    return identified_scenes  # Returning the identified scenes

"""
Generates a Tanka poem, following the 5-7-5-7-7 syllable structure.
"""
def create_tanka_poem(objects_in_video, scenes_in_video):
    top_three_scenes = get_three_scenes(scenes_in_video)
    scene_synonyms = {}
    for scene in top_three_scenes:
        # if is_noun(scene): # only parse words that are nouns
        synonyms = get_synonyms(scene)
        if synonyms: # don't add empty lists
            scene_synonyms[scene] = synonyms
    
    # dictionaries that maps syllable count to word (adjective, verbs, or adverbs)
    adjectives = create_syllable_dict('static/adjectives.txt')
    verbs = create_syllable_dict('static/verbs.txt')
    adverbs = create_syllable_dict('static/adverbs.txt')

    first_line = create_tanka_line(top_three_scenes[0], 5, adjectives, verbs, adverbs)
    second_line = create_tanka_line(top_three_scenes[1], 7, adjectives, verbs, adverbs)
    third_line = create_tanka_line(top_three_scenes[2], 5, adjectives, verbs, adverbs)

    if len(objects_in_video) >= 2:
        fourth_noun = objects_in_video[0]
        fifth_noun = objects_in_video[1]
    elif len(objects_in_video) == 1:
        fourth_noun = objects_in_video[0]
        selected_scene = random.choice(top_three_scenes)
        if selected_scene not in scene_synonyms:
            fifth_noun = selected_scene
        else:
            fifth_noun = random.choice(scene_synonyms[selected_scene])
    else:
        selected_scene = random.choice(top_three_scenes)
        if selected_scene not in scene_synonyms:
            fourth_noun = selected_scene
            fifth_noun = selected_scene
        else:
            fourth_noun = random.choice(scene_synonyms[selected_scene])
            fifth_noun = random.choice(scene_synonyms[selected_scene])
        
    fourth_line = create_tanka_line(fourth_noun, 7, adjectives, verbs, adverbs)
    fifth_line = create_tanka_line(fifth_noun, 7, adjectives, verbs, adverbs)

    tanka_poem = [first_line, second_line, third_line, fourth_line, fifth_line]
    return tanka_poem

"""
Creates a Tanka Line for the upper phrase via Tanka line formulas.
Takes in the noun, syllable count of
that line (5 or 7) and three dictionaries that maps the syllable count to either
list of adjective, verb, adverbs that fit that count.
"""
def create_tanka_line(noun, syllable_count, adjectives, verbs, adverbs):
    noun_syllables = count_syllables(noun)
    remaining_syllables = syllable_count - noun_syllables

    # Tanka formulas and their respective usage probabilities
    tanka_formulas = {
        "adjective_noun_verb": 20,
        "noun_adverb_verb": 20,
        "noun_verb": 15,
        "adjective_noun": 15,
        "noun_adjective": 10,
        "verb_noun": 10,
        "adverb_verb": 10
    }

    formula = random.choices(list(tanka_formulas.keys()), weights=tanka_formulas.values(), k=1)[0]

    noun = noun.lower()
    try:
        if formula == "adjective_noun_verb":
            for adj_syllables in range(1, remaining_syllables):
                verb_syllables = remaining_syllables - adj_syllables
                if adj_syllables in adjectives and verb_syllables in verbs:
                    adjective = random.choice(adjectives[adj_syllables]).lower().strip()
                    verb = random.choice(verbs[verb_syllables]).lower().strip()
                    return f"{adjective} {noun} {verb}"

        elif formula == "noun_adverb_verb":
            for adv_syllables in range(1, remaining_syllables):
                verb_syllables = remaining_syllables - adv_syllables
                if adv_syllables in adverbs and verb_syllables in verbs:
                    adverb = random.choice(adverbs[adv_syllables]).lower().strip()
                    verb = random.choice(verbs[verb_syllables]).lower().strip()
                    return f"{noun} {adverb} {verb}"

        elif formula == "noun_verb":
            if remaining_syllables in verbs:
                verb = random.choice(verbs[remaining_syllables]).lower().strip()
                return f"{noun} {verb}"

        elif formula == "adjective_noun":
            if remaining_syllables in adjectives:
                adjective = random.choice(adjectives[remaining_syllables]).lower().strip()
                return f"{adjective} {noun}"

        elif formula == "noun_adjective":
            if remaining_syllables in adjectives:
                adjective = random.choice(adjectives[remaining_syllables]).lower().strip()
                return f"{noun} {adjective}"

        elif formula == "verb_noun":
            if remaining_syllables in verbs:
                verb = random.choice(verbs[remaining_syllables]).lower().strip()
                return f"{verb} {noun}"
        
        elif formula == "adverb_verb":
            if remaining_syllables in adverbs:
                adverb = random.choice(adverbs[remaining_syllables]).lower().strip()
                verb = random.choice(verbs[remaining_syllables]).lower().strip()
                return f"{adverb} {verb}"
            
    except IndexError:
        pass

    return noun  # Return noun as fallback

"""
Given a word return its list of synonyms
"""
def get_synonyms(word):
    synonyms = set()

    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synoynm = lemma.name()
            if synoynm != word:
                synonyms.add(synoynm)  # Add the synonyms

    return list(synonyms)


"""
Helper Function to clear all the .jpg files in static/frames before each run
"""
def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

"""
Helper function that reads the category index files and returns a dict
"""       
def dict_index_to_class():
    class_index = {}

    with open('static/categories_hybrid1000.txt', 'r') as file:
        for line in file:
            if ',' not in line: # there is only one category
                if line.startswith('/'): # parses data in categories_places365
                    parts = line.split()
                    category = parts[0]
                    index = parts[1]
                else: # parses categories_hybrid1000 - parses single categories
                    parts = line.split()
                    index = parts[-1]
                    category = parts[1:-1]
                class_index[int(index)] = category
            else: # parses categories_hybrid1000 - parses multiple categories
                parts = line.strip().split(' ')
                index = int(parts[-1])  # The last part is the index
                categories = ' '.join(parts[1:-1])  # Joining all parts except the first (nXXXXXXXX) and the last (index)

                # Splitting categories by comma if there are multiple categories
                category_list = categories.split(', ') if ', ' in categories else [categories]
            
                class_index[int(index)] = category_list

    with open('static/categories_places365.txt', 'r') as file:
        for line in file:
            category, index = line.strip().split()
            class_index[int(index) + 1000] = category[3:]

    return class_index

"""
Helper function. Given a list of tuples -> ([places1, places2, ..], prob) 
return a list of the three scenes with the highest probability
"""
def get_three_scenes(scenes_in_video):
    scene_probabilities = {}  # Dictionary to store scenes and their probabilities

    # Iterate over all scenes and their probabilities
    for scenes, prob in scenes_in_video:
        for scene in scenes:
            if scene:  # Ensure the scene is not an empty string
                # Add or update the probability of the scene
                if scene in scene_probabilities:
                    scene_probabilities[scene] = max(scene_probabilities[scene], prob)
                else:
                    scene_probabilities[scene] = prob

    # Sort scenes by their probabilities in descending order and get top 3
    top_three_scenes = sorted(scene_probabilities, key=scene_probabilities.get, reverse=True)[:3]
    return top_three_scenes

"""
Helper Function. Checks if a word is noun via nltk pos tagging
"""
def is_noun(word):
    pos = nltk.pos_tag([word])[0][1]  # POS tag the word and get the tag
    # The POS tags starting with 'NN' represent different forms of nouns
    return pos in ["NN", "NNS", "NNP", "NNPS"]


""" 
Helper Function. Returns the number of syllables in a word. If the word is not found,
a heuristic is used to approximate the number of syllables. 
"""
def count_syllables(word):
    d = cmudict.dict()
    word = word.lower()
    if word in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        # Fallback heuristic for words not in the CMU dictionary
        count = 0
        vowels = 'aeiouy'
        word = word.lower()
        if word[0] in vowels:
            count +=1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count+=1
        if count == 0:
            count +=1
        return count

"""
Helper Function that returns a dictionary of the syllable count to the words
with that syllable count
"""
def create_syllable_dict(filename):
    syllable_dict = {}
    with open(filename, 'r') as file:
        for word in file:
            word = word.strip()
            syllable_count = count_syllables(word)
            if syllable_count not in syllable_dict:
                syllable_dict[syllable_count] = [word]
            else:
                syllable_dict[syllable_count].append(word)
    return syllable_dict

"""
Saves the poem to a file so that the user can listen to it again.
"""
def save_poem(poem_data):
    # Append poem data to a file
    with open('saved_poems.json', 'a') as file:
        json.dump(poem_data, file)
        file.write("\n")  # Newline for separating JSON objects


"""
Function to load all the saved poems onto index.html
"""
def load_poems():
    poems = []
    try:
        if os.path.exists('saved_poems.json'):
            with open('saved_poems.json', 'r') as file:
                for line in file:
                    poem_data = json.loads(line.strip())
                    poems.append(poem_data)
    except Exception as e:
        print(f"Error loading poems: {e}")
    return poems

"""
Evaluation function to check how many tanka lines have the correct syllables.
"""
def evaluate_tanka_poem(tanka_poem):
    # Expected syllable structure of a tanka poem
    expected_syllables = [5, 7, 5, 7, 7]
    
    # Check if the poem has the correct number of lines
    if len(tanka_poem) != 5:
        return f"Incorrect number of lines. Expected 5, found {len(tanka_poem)}"

    # Evaluate each line
    correct_lines = 0
    for line, expected in zip(tanka_poem, expected_syllables):
        syllable_count = count_syllables(line)
        if syllable_count == expected:
            correct_lines += 1
        else:
            print(f"Line '{line}' has {syllable_count} syllables, expected {expected}.")

    return f"{correct_lines}/5 lines have the correct syllable count."

if __name__ == "__main__":
    clear_folder("static/frames")
    app.run(debug=True)