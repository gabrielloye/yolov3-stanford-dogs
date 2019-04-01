# Object Detection and Image Classification of Stanford Dogs Dataset # 

#### Object Detection ####
[YOLOv3](https://github.com/ultralytics/yolov3) model was trained [here](https://github.com/gabrielloye/yolov3-stanford-dogs/blob/master/train.ipynb) using the annotations to identify the location of a single class(dog) in an image.

#### Image Classification ####
Transfer Learning was used to train the classifier of the InceptionV3 model to identify the breed of the dogs apart from the 119 other dog breeds.
[Link to Kaggle Kernel](https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation)

#### Bringing them together ####
[Main](https://github.com/gabrielloye/yolov3-stanford-dogs/blob/master/main.ipynb) notebook shows how the 2 models can be brought together to process an image and identify the dog breed.
