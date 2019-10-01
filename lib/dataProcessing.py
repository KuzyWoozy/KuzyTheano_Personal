import mnist

import numpy as np
import dill as pickle


def LoadTextLines(file2load):
    with open(file2load, "r") as fileLoaded:
        return fileLoaded.read().strip().split()


def LoadData(dataFolder):
    if dataFolder == "mnist_train":
        trainInput = np.array(mnist.train_images(), dtype="float32")
        trainInput  = trainInput.reshape(len(trainInput), 784)/np.max(trainInput)
        

        trainOutput = mnist.train_labels()
        trainInput_max = np.max(trainOutput)
        trainInput_min = np.min(trainOutput)

        trainOutput = np.array([[1 if iteration==y else 0 for iteration in range(int(trainInput_max-trainInput_min)+1)] for y in trainOutput], dtype="float32")

        return trainInput, trainOutput


    if dataFolder == "mnist_test":
        testInput = np.array(mnist.test_images(), dtype="float32")
        testInput = testInput.reshape(len(testInput), 784)/np.max(testInput)


        testOutput = mnist.test_labels()

        testInput_max = np.max(testOutput)
        testInput_min = np.min(testOutput)

        testOutput = np.array([[1 if iteration==y else 0 for iteration in range(int(testInput_max-testInput_min)+1)] for y in testOutput], dtype="float32")

        return testInput, testOutput

    else:
        return np.array([[eval(item)] for item in LoadTextLines("{}/input.txt".format(dataFolder))], dtype="float32"), np.array([[eval(item)] for item in LoadTextLines("{}/desire.txt".format(dataFolder))], dtype="float32")






def ResultFormat(**kwargs):
    return """{prettyLine}
Network Structure: {networkDesign}
rValue: {rValue}
momentum: {momentumValue}
cost: {costMethod}
weightInit: {weightInit}
Train accuracy: {trainAccuracy}
Test accuracy: {testAccuracy}\n""".format(prettyLine="".join(["-" for _ in range(35)]),
                           networkDesign=kwargs["networkDesign"],
                           rValue=kwargs["rValue"],
                           momentumValue=kwargs["momentumValue"],
                           costMethod=kwargs["costMethod"],
                           paramInitMethod=kwargs["paramInitMethod"],
                           trainAccuracy=kwargs["trainAccuracy"],
                           testAccuracy=kwargs["testAccuracy"]
                           )


def SaveText(file2Save2, text):
    with open(file2Save2,"a") as file2Save2:
        file2Save2.write(text)


def LoadNetwork(network2Load):
    with open(network2Load,"rb") as networkPickled:
        return pickle.load(networkPickled)