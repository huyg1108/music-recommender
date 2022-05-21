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
* We use [Convolution neural network (CNN) using tensorflow](https://www.tensorflow.org/tutorials/images/cnn) to build this model
#### Some problems: 
* The first version of this model is lag on my device while running app, so I must simplify this and I didn't use any pre-trained model. It is why this model has low accuracy.
* It needs to be fine-tuned for better result.
* Can try Vision Transformer instead of using CNN for better result.

## Installation
#### Clone this repository: 
$ git clone https://github.com/huyg1108/music_recommender.git

#### Run app:
$ cd music_recommender <br/>
$ pip install -r requirements.txt <br/>
$ python3 app.py

## Authors:
- **Triệu Gia Huy** - [huyg](https://github.com/huyg1108)
