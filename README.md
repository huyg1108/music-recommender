# Music recommender based on face emotion

This app will recommend music based on your face, it will detect your face and predict your emotion, and then this app will give you music for your mood.

## Dataset
Training data and testing data that were used are sourced from kaggle: https://www.kaggle.com/datasets/msambare/fer2013 <br/> The training set consists of 28,709 examples and the testing set consists of 3,589 examples.

Dataset       | Directories     | Files
------------- | -------------   | -------------
Test          | angry           | 958
|             | disgust         | 111
|             | fear            | 1024
|             | happy           | 1774
|             | neutral         | 1233
|             | sad             | 1247
|             | surprise        | 831
Train         | angry           | 3995
|             | disgust         | 436
|             | fear            | 4097
|             | happy           | 7215
|             | neutral         | 4965
|             | sad             | 4830
|             | surprise        | 3171

## About detection model
* I use [Convolution neural network (CNN) using tensorflow](https://www.tensorflow.org/tutorials/images/cnn) to build this model
* Pre-trained model:
- [Model structure](https://github.com/serengil/deepface/blob/master/deepface/extendedmodels/Emotion.py)
- [Pre-trained weight](https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5)
* Note: this pre-trained model is not good so I trained it again with 50 epochs and fine-tuned it with 6 trainable layers and 10 epochs

## Installation
#### Clone this repository: 
$ git clone https://github.com/huyg1108/music-recommender.git

#### Run app:
$ cd music-recommender <br/>
$ pip install -r requirements.txt <br/>
$ python3 app.py

## Authors:
- **Triệu Gia Huy** - [huyg](https://github.com/huyg1108)
