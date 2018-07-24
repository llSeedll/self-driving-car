## Self Driving Car

This project provides accurate prediction of steering angles of self driving cars. It was inspired by [Udacity's Self driving car Nanodegree](https://github.com/udacity/CarND-Behavioral-Cloning-P3) and NVIDIA's [End to End Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

The code uses Convolutional Neural Networks are used to predict the streering angle corresponding to an input image of the road.

### Requirements
To install the requirements, execute from your working directory, the following code in the terminal.

##### `pip install requirements.txt`

### Dataset
The dataset can be downloaded [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) and should be extracted into the dataset folder at the root of the repository folder

### Preprocessed files & train model
You can download the already preprocessed files and use them directly to train the model:<br/>

* [labels](https://drive.google.com/open?id=13Y8OPTgiuxxTbcHE1H8OW5GboC8ppyxL)
* [preprocessed](https://drive.google.com/open?id=1Fk6Uq-MF_SbmSpvaPfKc1fJbolLO_6Xj)
	
If the pretrained model `model_adam_mse.h5` is used, only the step 4 of the procedure bellow will be left to execute.


### Procedure

1) Make sure the dataset is in the `dataset` folder.
2) Run `load_preprocessed_data.py` to get dataset from the folder and store it the preprocessed version in a pickle file.
3) Run `train.py` to load the preprocessed data from the previously savec pickle file and perform the training operations.
4) Run `self_driving_car.py` to test your results on the video.

<img src="https://github.com/llSeedll/self-driving-car/blob/master/capture.gif">

### References:
 
 - [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) 
 - Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
 - Inspired from this github repository: https://github.com/SullyChen/Autopilot-TensorFlow

