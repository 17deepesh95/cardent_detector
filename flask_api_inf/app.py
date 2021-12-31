import csv
import os
import time

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

BASE_DIR = 'E:/projects/cardent_detector'
FLASK_DIR = os.path.join(BASE_DIR, 'flask_api_inf')
MODEL_DIR = os.path.join(BASE_DIR, 'models_output')

MODEL_FILE = os.path.join(MODEL_DIR, 'csv_retinanet_25.pt')
CLASSESS_FILE = os.path.join(BASE_DIR, 'input_csv', 'classes.csv')

app = Flask(__name__)


def load_model(model_path):
    model = torch.load(model_path)
    return model


loaded_model = load_model(MODEL_FILE)


@app.route('/')
def hello_world():
    return 'This is my first API call'


@app.route('/detect_dent', methods=["POST"])
def detect_cat_dent():
    image = request.files['image']
    read_image = Image.open(image)
    read_image = read_image.convert("RGB")

    try:
        os.mkdir("temp_image_store")
    except FileExistsError:
        pass

    read_image.save("temp_image_store/temp_image.jpg")

    return_json = predict_image(loaded_model)

    return jsonify(return_json)


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def read_labels(classes_csv):
    with open(classes_csv, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    return labels


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def predict_image(model, image_path='temp_image_store/temp_image.jpg'):
    """

    :param model: loaded model
    :param image_path: path to the image to infer
    :return: json with information
    """

    return_dict = {}
    labels = read_labels(CLASSESS_FILE)
    image = cv2.imread(image_path)

    if image is None:
        return 0

    image_orig = image.copy()

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - rows % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]

    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))
    predictions = []
    with torch.no_grad():
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
        st = time.time()
        print(image.shape, image_orig.shape, scale)
        scores, classification, transformed_anchors = model(image.cuda().float())

        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)

            label_name = labels[int(classification[idxs[0][j]])]
            score = scores[j]

            caption = '{} {:.3f}'.format(label_name, score)

            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            temp_dict = {
                'label': label_name,
                'score': float(score),
                'bbox': [x1, y1, x2, y2]
            }
            predictions.append(temp_dict)

    return_dict = {'imageName': image_path, 'predictions': predictions}

    return return_dict
