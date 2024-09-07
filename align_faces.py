import os
import cv2
import json
import pathlib
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from insightface.app import FaceAnalysis
import requests
import torch
from PIL import Image

# Create the output directory if it doesn't exist
output_dir = "json_maps"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [29]  # 29 corresponds to 'hat' in the COCO dataset annotations

# Function to resolve relative paths
def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

# Function to detect the dominant color in a region
def detect_hex_color(region_img):
    if region_img.size == 0:
        return "NONE", 0.0

    region_img = cv2.resize(region_img, (50, 25))
    region_img = region_img.reshape((region_img.shape[0] * region_img.shape[1], 3))

    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(region_img)
    label_counts = Counter(labels)

    most_common_label, count = label_counts.most_common(1)[0]
    dominant_color = kmeans.cluster_centers_[most_common_label]

    total_count = sum(label_counts.values())
    certainty = (count / total_count) * 100

    dominant_color_rgb = dominant_color[::-1]
    dominant_color_hex = '#%02x%02x%02x' % (int(dominant_color_rgb[0]), int(dominant_color_rgb[1]), int(dominant_color_rgb[2]))

    return dominant_color_hex, certainty

def get_color_name_from_api(hex_color):
    response = requests.get(f"https://www.thecolorapi.com/id?hex={hex_color.lstrip('#')}")
    if response.status_code == 200:
        color_data = response.json()
        return color_data['name']['value']
    return "unknown"

def get_eye_regions(image, landmarks):
    left_eye_coords = landmarks["eyes"]["left_eye"]
    right_eye_coords = landmarks["eyes"]["right_eye"]

    left_eye_region = image[max(0, int(left_eye_coords[1]) - 10):min(image.shape[0], int(left_eye_coords[1]) + 10),
                            max(0, int(left_eye_coords[0]) - 10):min(image.shape[1], int(left_eye_coords[0]) + 10)]
    right_eye_region = image[max(0, int(right_eye_coords[1]) - 10):min(image.shape[0], int(right_eye_coords[1]) + 10),
                             max(0, int(right_eye_coords[0]) - 10):min(image.shape[1], int(right_eye_coords[0]) + 10)]

    return left_eye_region, right_eye_region

def get_facial_hair_region(image, landmarks):
    mouth_left_corner = landmarks["mouth"]["left_corner"]
    mouth_right_corner = landmarks["mouth"]["right_corner"]

    facial_hair_region = image[max(0, int(mouth_left_corner[1]) - 20):min(image.shape[0], int(mouth_left_corner[1]) + 20),
                               max(0, int(mouth_left_corner[0]) - 20):min(image.shape[1], int(mouth_right_corner[0]) + 20)]

    return facial_hair_region

def get_head_hair_region(image, landmarks):
    eye_left = int(landmarks["eyes"]["left_eye"][0])
    eye_right = int(landmarks["eyes"]["right_eye"][0])
    nose_bottom = int(landmarks["nose"][1])

    hair_start = max(0, nose_bottom - 60)
    hair_end = nose_bottom - 20
    hair_region = image[hair_start:hair_end, eye_left:eye_right]

    return hair_region

def detect_headwear(image):
    results = model(image)
    labels, confidences = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2]

    headwear_detected = False
    headwear_confidence = 0.0

    for label, confidence in zip(labels, confidences):
        if label == 29:
            headwear_detected = True
            if confidence > headwear_confidence:
                headwear_confidence = float(confidence)

    headwear_confidence = headwear_confidence * 100
    return headwear_detected, headwear_confidence

def generate_face_profile(image_path: str) -> dict:
    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image at {image_path}")

    faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in the image.")

    face = faces[0]
    embedding = face.embedding.tolist()

    landmarks = {
        "eyes": {
            "left_eye": face.landmark_2d_106[36].tolist(),
            "right_eye": face.landmark_2d_106[45].tolist()
        },
        "nose": face.landmark_2d_106[30].tolist(),
        "mouth": {
            "left_corner": face.landmark_2d_106[48].tolist(),
            "right_corner": face.landmark_2d_106[54].tolist()
        }
    }

    left_eye_region, right_eye_region = get_eye_regions(img, landmarks)
    left_eye_color, left_eye_certainty = detect_hex_color(left_eye_region)
    right_eye_color, right_eye_certainty = detect_hex_color(right_eye_region)

    eye_color_guess1 = get_color_name_from_api(left_eye_color)
    eye_color_guess2 = get_color_name_from_api(right_eye_color)

    print(f"Left eye color: {left_eye_color} ({eye_color_guess1}) ({left_eye_certainty:.2f}% certain)")
    print(f"Right eye color: {right_eye_color} ({eye_color_guess2}) ({right_eye_certainty:.2f}% certain)")

    facial_hair_region = get_facial_hair_region(img, landmarks)
    facial_hair_color, facial_hair_certainty = detect_hex_color(facial_hair_region)
    facial_hair_color_name = get_color_name_from_api(facial_hair_color)

    print(f"Facial hair color: {facial_hair_color} ({facial_hair_color_name}) ({facial_hair_certainty:.2f}% certain)")

    head_hair_region = get_head_hair_region(img, landmarks)
    head_hair_color, head_hair_certainty = detect_hex_color(head_hair_region)
    head_hair_color_name = get_color_name_from_api(head_hair_color)

    print(f"Head hair color: {head_hair_color} ({head_hair_color_name}) ({head_hair_certainty:.2f}% certain)")

    headwear_detected, headwear_confidence = detect_headwear(img)
    if headwear_detected:
        headwear_region = get_headwear_region(img, landmarks)
        if headwear_region.size > 0:
            headwear_color, headwear_color_certainty = detect_hex_color(headwear_region)
            headwear_color_name = get_color_name_from_api(headwear_color)
        else:
            headwear_color = "NONE"
            headwear_color_name = "NONE"
            headwear_color_certainty = 0.0

        print(f"Headwear color: {headwear_color} ({headwear_color_name}) ({headwear_color_certainty:.2f}% certain)")
    else:
        headwear_color = "NONE"
        headwear_color_name = "NONE"
        headwear_confidence = 100.0 - headwear_confidence

        print(f"Headwear detected: {headwear_detected} (confidence: {headwear_confidence:.2f}%)")

    profile = {
        "reference_images": [
            {"pose": "front", "embedding": embedding}
        ],
        "facial_landmarks": landmarks,
        "physical_features": {
            "eye_color1": left_eye_color,
            "eye_color2": right_eye_color,
            "eye_color_guess1": eye_color_guess1,
            "eye_color_guess2": eye_color_guess2,
            "facial_hair": {
                "type": "unknown",
                "color": facial_hair_color,
                "color_guess": facial_hair_color_name,
                "certainty": facial_hair_certainty
            },
            "head_hair": {
                "color": head_hair_color,
                "color_guess": head_hair_color_name,
                "certainty": head_hair_certainty
            },
            "headwear": {
                "detected": headwear_detected,
                "color": headwear_color,
                "color_guess": headwear_color_name,
                "confidence": headwear_confidence
            },
            "skin_tone": "unknown"
        },
        "accessories": {
            "hat": {"type": "unknown", "color": headwear_color_name},
            "glasses": {"type": "unknown", "color": "unknown"},
            "earrings": "unknown"
        },
        "clothing": {
            "upper_body": {"type": "unknown", "color": "unknown", "pattern": "unknown"},
            "lower_body": {"type": "unknown", "color": "unknown"}
        },
        "contextual_features": {
            "height": "unknown",
            "build": "unknown",
            "movement": "unknown"
        }
    }

    return profile

def process_images(image_paths: list):
    for image_path in image_paths:
        try:
            profile = generate_face_profile(image_path)
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            json_filename = f"{name}_{ext[1:].replace('.', '_')}.json"
            json_filepath = os.path.join(output_dir, json_filename)

            with open(json_filepath, "w") as json_file:
                json.dump(profile, json_file, indent=4)
            print(f"Profile for {image_path} saved to {json_filepath}")
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

image_list = ["users/andy.png"]
process_images(image_list)