# Minimal Facial Recog

Minimal Facial Recog is a Python-based application designed to analyze facial images, extract detailed facial features, and generate a profile with an array of physical and contextual attributes. This tool employs multiple libraries and models to identify and evaluate facial landmarks, detect colors, and estimate the presence of accessories such as hats.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Acknowledgments](#acknowledgments)

## Requirements

- Python 3.7+
- `cv2` (OpenCV)
- `numpy`
- `scikit-learn`
- `requests`
- `torch`
- `Pillow`
- `insightface`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourrepo/minimal-facial-recog.git
    cd minimal-facial-recog
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python-headless numpy scikit-learn requests torch pillow insightface
    ```

3. We also rely on the YOLOv5 model from ultralytics to detect headwear:
    ```sh
    pip install git+https://github.com/ultralytics/yolov5.git
    ```

## Usage

1. Place your images in the `users` directory (or adjust the `image_list` accordingly).

2. Run the script:
    ```sh
    python minimal_facial_recog.py
    ```

## Output

The script will generate JSON files in the `json_maps` directory with detailed facial profiles. Each JSON contains information such as:

- Facial landmarks
- Eye, hair and facial hair colors (with detection certainty)
- Headwear detection (if any)
- Physical features and contextual features placeholder

### Example Output JSON

```json
{
  "reference_images": [
    {"pose": "front", "embedding": [/* embeddings */]}
  ],
  "facial_landmarks": {
    "eyes": {"left_eye": [x, y], "right_eye": [x, y]},
    "nose": [x, y],
    "mouth": {"left_corner": [x, y], "right_corner": [x, y]}
  },
  "physical_features": {
    "eye_color1": "#RRGGBB",
    "eye_color2": "#RRGGBB",
    "eye_color_guess1": "Blue",
    "eye_color_guess2": "Blue",
    "facial_hair": {"type": "unknown", "color": "#RRGGBB", "color_guess": "Brown", "certainty": 85.5},
    "head_hair": {"color": "#RRGGBB", "color_guess": "Black", "certainty": 90.0},
    "headwear": {"detected": true, "color": "#RRGGBB", "color_guess": "Red", "confidence": 99.0},
    "skin_tone": "unknown"
  },
  "accessories": {
    "hat": {"type": "unknown", "color": "Red"},
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
```

## Acknowledgments

- **InsightFace**: For providing state-of-the-art face analysis.
- **Ultralytics YOLOv5**: For headwear detection.
- **The Color API**: For converting hex colors to human-readable color names.

Feel free to contribute, report issues, or fork the repository!