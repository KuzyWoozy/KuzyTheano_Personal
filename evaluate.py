import dill as pickle
import numpy as np
import argparse as arg

from lib import dataProcessing


argParser = arg.ArgumentParser(description="Evaluate trained networks")
argParser.add_argument("networkFile", metavar="PICKLE_FILE", type=str, help="The network to evaluate")
argParser.add_argument("-l", "--load", metavar="DATA", dest="data", type=str, default="mnist_test", help="The data to use")
#  ["models/FNN.pickle"]
args = argParser.parse_args()


network = dataProcessing.LoadNetwork(args.networkFile)

inputData, desiredData = dataProcessing.LoadData(args.data)

correct = 0
for x, desire in zip(inputData, desiredData):
   
    if network.Predict(x) == np.argmax(desire):
        correct += 1

print("Accuracy of: {}%".format((correct/len(desiredData)) * 100))






