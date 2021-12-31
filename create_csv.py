import pandas as pd
import xml.etree.ElementTree as ET
import os

folder = 'dataset/car_dent_voc/valid_xml'

files = os.listdir(folder)

output_dict = []
unique_label = []

for file in files:
    print(file)
    file_path = os.path.join(folder, file)

    tree = ET.parse(file_path)

    root = tree.getroot()

    annotations = root.find('annotation')
    file_name = root.find('filename').text

    obj = root.findall('object')

    for o in obj:
        label = o.find('name').text
        bbox = o.find('bndbox')

        x_min = bbox.find('xmin').text
        y_min = bbox.find('ymin').text
        x_max = bbox.find('xmax').text
        y_max = bbox.find('ymax').text

        dictionary = {
            'file_name': file_name,
            'x1': x_min,
            'y1': y_min,
            'x2': x_max,
            'y2': y_max,
            'label': label
        }
        output_dict.append(dictionary)

        if label not in unique_label:
            unique_label.append(label)

# label_list = []
# fi = open('input_csv/classes.csv', 'a')
# for i, lbl in enumerate(unique_label):
#     fi.write(f'{lbl},{i}\n')
#
# fi.close()

import json

# json.dump(label_list, open('input_csv/classes.json', 'w'))
#
# label_dataset = pd.read_json('input_csv/classes.json')
# label_dataset.to_csv('input_csv/classes.csv', header=False)

json.dump(output_dict, open('input_csv/val_csv.json', 'w'))
dataset = pd.read_json('input_csv/val_csv.json')
dataset.to_csv('input_csv/val_csv.csv', index=False, header=False)

