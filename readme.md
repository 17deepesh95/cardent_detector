## Retinanet Implementation in Pytorch for detecting dents and damage on Car images.
## _V - 1.0_

* _Added Modules for training with retinaNet [1]_
* _Added Modules for testing the trained models_

## _V - 1.1_
* _Added Flask app for api calls for image inference._

## System Information
1. #### Python-3.7
2. #### Pytorch-1.10.1
3. #### Training GPU - NVIDIA RTX 2060

## Usage
Download model from <a href="#"> here </a> (Used ResNet101 [2] as backbone for the training.
1. Recommended to create a new virtual environment.
2. Install dependencies by _"pip install -r requirements.txt"_
3. To train a model from scratch
   1. make sure train and val csv are created with following format
      1. _**file_path,x1,y1,x2,y2,label**_
      2. the csv should not have any headers and index
         (There can be multiple records for single image with different labels)
   3. make sure classes csv is created for class mapping with following format
      1. _**class_name,index**_
      2. the csv should not have any headers and index
   4. use the following command to train the model
      1. _python train.py --dataset csv --csv_train \<path to train csv> --csv_classes \<path to classes csv> --csv_val \<path to validation csv> --depth 101_
         1. **--dataset** = type of dataset input can be csv or default coco dataset (csv/coco)
         2. **--csv_train** = input dataset with train annotations
         3. **--csv_classes** = input csv with class mapping
         4. **--csv_val** = input dataset with validation annotations
         5. **--depth** = resnet depth (18/34/50/101/152)
   5. Models will be stored after every epoch **TODO: add early stopping.**
4. To test the model for every image in a folder run the following command.
   1. _python visualize_single_image --image_dir \<path to directory with test imaages> --model_path \<path to trained model> --class_list \<path to csv with class mapping>_

## API Usage
1. To run flask app navigate to "flask_api_inf" and run - 
   1. _flask app.py_
2. The flask app will run on **"http://127.0.0.1:5000"**
3. Use postman to send a POST request to **http://127.0.0.1:5000/detect_dent** with following arguments.
      1. Body - 
         1. image (type-file) - select the image file.
4. The api will return a Json 
   1. example json - 
      _{
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
}_

### References
1. https://arxiv.org/abs/1708.02002 - Focal Loss for Dense Object 
    Detection _by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar_
2. https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition _by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun_