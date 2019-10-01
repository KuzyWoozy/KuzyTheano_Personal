import dill as pickle
import numpy as np
import argparse as arg

from lib import dataProcessing
from PIL import Image
from PIL import ImageOps


argParser = arg.ArgumentParser(description="Evaluation of networks")
argParser.add_argument("networkFile", metavar="FILE", type=str, help="The network to use")
argParser.add_argument("image", metavar="IMG", type=str, help="The image to predict")
argParser.add_argument("-i", "--invert", action="store_true", help="Invert the colours of the image")
args = argParser.parse_args()

def loadImage(imageFile):
    with Image.open(imageFile).convert("L") as img:
        if args.invert == True:
            img = ImageOps.invert(img).resize((28,28))
        if args.invert == False:
            img = img.resize((28,28))
        
        img.save("tmp/IMG_TESTING.png")
        return np.resize(list(img.getdata()), (784,)).astype("float32")/np.max(list(img.getdata()))


network = dataProcessing.LoadNetwork(args.networkFile)

testImage = loadImage(args.image)

print("Network prediction is {}".format(network.Predict(testImage)))



