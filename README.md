#  Feed forward neural networks via Theano

Note that by default these networks can be trained on mnist data, but has minimal support
for training on specified data (see train.py -l option for details)

**This has been created using Python 3.7.4, on Linux 5.2.8-1-MANJARO**

libraries required (please install via pip):
  - numpy
  - pillow
  - theano
  - mnistx
  - dill

## Overview

All the .py files in the main directory are interacted with via command line by specifying
various arguments, like you do with shell commands on Linux:


Example:
    python train.py -h

        This will show all the options which can be used with the train.py program
    

There are 3 main .py files used when interacting with the feed forward neural networks:

    train.py:
        - Used to train new models or resume training on already made models
        - Can specify various configurations for the neural networks, such as
        the cost function used, STG step, momentum step, etc... (use -h to see
        available options)

    evaluate.py:
        -  Used to test the trained models, giving them percentage scores for their
        accuracy in processing of the supplied test data.

    predict.py:
        -  Used to process images and have the specified trained model 
        -  Please note that some images will have to be inverted in terms
        of colours in order for the network to understand them. Networks
        see white as input, anything lower is an input of lower intensity.


## Getting started

# Minimum:
    The minmum things you can run to train and test a neural network (Using the mnist data set)

    python train.py -nD "20,leakyrelu|20,leakyrelu|softmax_s"
    python evaluate models/FNN.pickle


# TRAINING:
    In order to train a neural network, it has to be designed via the train.py

    Example:
        python train.py -nD "100,leakyrelu|60,leakyrelu|softmax_s"
            This will create a neural network with 3 layers, with the specified sizes of neurones,
            note that the number of neurones for the last layer will be determined automatically.
        
        python train.py -nD "100,leakyrelu|60,leakyrelu|softmax_s" -c "mse" -r "0.0001"
            Will create the same network, but with a different cost function STG step value
       
        python train.py -nD "100,leakyrelu|90,sigmoid|80,leakyrelu|70,tanh|softmax_s"
            5 layered neural network 
        

        USE -h OPTION TO SEE MORE CONFGIGURABLE OPTIONS

    Good models to try:
        python train.py -nD "20,leakyrelu|20,leakyrelu|softmax_s" -r "0.0001"
        python train.py -nD "20,leakyrelu|20,leakyrelu|sigmoid" -r "0.01" -m "0" -c "mse"
        python train.py -nD "60,leakyrelu|60,leakyrelu|60,leakyrelu|softmax_s" -r "0.0001" -m "0.7"
        python train.py -nD "20,leakyrelu|20,leakyrelu|sigmoid" -r "0.0001"
        python train.py -nD "100,leakyrelu|100,leakyrelu|softmax_s" -e "3" -r "0.0001"



# EVALUATION:
    After you have trained the network, you can see what scores it gets on the training data.

    Example:
        python evaluate.py PATH_TO_NETWORK_FILE -l test_data
            This will evaluate the network specified by PATH_TO_NETWORK_FILE, using the data specified by -l
        


# PREDICTION:
    ONLY SUPPORTED FOR CLASSIFICATION OF NUMBERS, aka images of numbers (some examples in 'tmp' directory)

    Once the network has been trained and has good evaluation accuracy, it can be
    tested on images of numbers to see if your network gets them right. Note that
    some images withh have to be inverted via the "-i" option in order to be viewed 
    by the network correctly

    Example:
        python predict.py PATH_TO_NETWORK_FILE PATH_TO_IMAGE [-i]
            Predict the number shown on the image, via the specified network and image
            to attempt to predict the number from







