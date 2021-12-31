# Retinanet Implementation in Pytorch for detecting dents and damage on Car images.
## V - 1.0
<ul>
<li>Added Modules for training with retinaNet [1]</li>
<li>Added Modules for testing the trained models</li>
</ul>

## V - 1.1
* Added Flask app for api calls for image inference.

## System Information
1. Python-3.7
2. Pytorch-1.10.1
3. Training GPU - NVIDIA RTX 2060

## Usage
Download model from <a href="#"> here </a> (Used ResNet101 as backbone for the training.
1. Recommended to create a new virtual environment.
2. Install dependencies by "pip install -r requirements.txt"
3. To train a model from scratch
   1. make sure train and val csv are created with following format
      1. **file_path,x1,y1,x2,y2,label**
      2. the csv should not have any headers and index
         (There can be multiple records for single image with different labels)
   3. make sure classes csv is created for class mapping with following format
      1. **class_name,index**
      2. the csv should not have any headers and index
   4. use the following command to train the model
      1. python train.py --dataset csv --csv_train \<path to train csv> --csv_classes \<path to classes csv> --csv_val \<path to validation csv> --depth 101
         1. --dataset = type of dataset input can be csv or default coco dataset (csv/coco)
         2. --csv_train = input dataset with train annotations
         3. --csv_classes = input csv with class mapping
         4. --csv_val = input dataset with validation annotations
         5. --depth = resnet depth (18/34/50/101/152)
   5. Models will be stored after every epoch **TODO: add early stopping.**
4. To test the model for every image in a folder run the following command.
   1. python visualize_single_image --image_dir \<path to directory with test imaages> --model_path \<path to trained model> --class_list \<path to csv with class mapping>

## API Usage
1. To install dependencies install flask_requirements.txt along with requirements.txt
2. To run flask app navigate to "flask_api_inf" and run - 
   1. flask app.py
3. The flask app will run on "http://127.0.0.1:5000"
4. Use postman to send a POST request to http://127.0.0.1:5000/detect_dent with following arguments.
   1. Body - 
      1. image (type-file) - select the image file.
5. The api will return a Json
   1. example json
      1. {
    "imageName": "temp_image_store/temp_image.jpg",
    "predictions": [
        {
            "bbox": [
                12,
                25,
                389,
                569
            ],
            "label": "fender-dent",
            "score": 0.2705020308494568
        }
    ]
}

### References
1. <a href="https://arxiv.org/abs/1708.02002">
        https://arxiv.org/abs/1708.02002
    </a> - Focal Loss for Dense Object 
    Detection by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar
