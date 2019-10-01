import mnist
import time
import sys

import theano as th
import argparse as arg
import numpy as np

from lib import neural 
from lib import cost
from lib import paramGen
from lib import dataProcessing


sys.setrecursionlimit(2000)

argParser= arg.ArgumentParser(description=
                        """
                        Train the neural network, specifying the desired params and options
                        """,
                        formatter_class=arg.ArgumentDefaultsHelpFormatter)


netGroup = argParser.add_mutually_exclusive_group(required=True)

netGroup.add_argument("-n", "--network", metavar="NETWORK_FILE", dest="networkFile", type=str, help="Specify the network to load")
netGroup.add_argument("-nD", "--networkDesign", metavar="NETWORK_DESIGN", dest="networkDesign", type=str, help="Specify the design of our feed-forward neural network")

argParser.add_argument("-l", "--load", metavar="DATA", dest="data", type=str, default="mnist_train", help="Folder to load the data from, needs to have a input.txt and desire.txt to get data from")
argParser.add_argument("-p", "--paramInit", metavar="PARAM_INIT_METHOD", dest="paramInitMethod", type=str, default="normal", choices=["normal"], help="Parameter generation technique")
argParser.add_argument("-c", "--cost", metavar="COST_METHOD", dest="costMethod", type=str, default="crossentropy", choices=["crossentropy", "mse"], help="Supported cost functions")
argParser.add_argument("-r", "--rValue", metavar="R_VALUE", dest="rValue", type=float, default=0.0001, help="Step value for STG")
argParser.add_argument("-m", "--momentum", metavar="MOMENTUM_VALUE", dest="momentumValue", type=float, default=0.95, help="Momentum value for STG")
argParser.add_argument("-s", "--save", metavar="FILE", dest="saveFile", type=str, default="models/FNN.pickle", help="File to save the network to")
argParser.add_argument("-e", "--epochs", metavar="NUM_EPOCHS", type=int, default=1, help="Number of epochs to use when training network")

args = argParser.parse_args()

inputData, desiredData = dataProcessing.LoadData(args.data)

if args.networkFile == None:
    input = ",".join([str(inputData.shape[1]), "input"])
    hidden = "|".join(args.networkDesign.split("|")[:-1])
    output = ",".join([str(desiredData.shape[1]),args.networkDesign.split("|")[-1]])

    networkShape = "|".join([input,hidden,output])
    network = neural.NetworkCreator(networkShape.lower(), paramGen.paramGenerators[args.paramInitMethod.lower()](), cost.costFuncs[args.costMethod.lower()] , args.rValue, args.momentumValue)
else:
    network = dataProcessing.LoadNetwork(args.networkFile)


try:
    for _ in range(args.epochs):
        network.Propogate(inputData, desiredData)
finally:
    network.Save(args.saveFile)





