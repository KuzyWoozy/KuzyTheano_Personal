import theano as th
import numpy as np


class LeakyRelu():
    def __init__(self, layerZGraph, layerPosition=None):
        self.name = "leakyrelu (layer: {0})".format(layerPosition)
        self.negativeConstant = 0.01

        """
        Function
        """
        self.activation_graph = th.tensor.set_subtensor(layerZGraph[layerZGraph<0], layerZGraph[layerZGraph<0] * self.negativeConstant)
        
        """
        Derivitive
        """
        layerZGraph = th.tensor.set_subtensor(layerZGraph[layerZGraph>0], 1)
        self.activationDerv_graph = th.tensor.set_subtensor(layerZGraph[layerZGraph<0], self.negativeConstant)
        

class Sigmoid():
    def __init__(self, layerZGraph, layerPosition=None):
        self.name = "sigmoid (layer: {0})".format(layerPosition)
    
        """
        Function
        """
        self.activation_graph = 1/(1+th.tensor.exp(-layerZGraph))
       
        """
        Derivitive
        """
        self.activationDerv_graph = self.activation_graph * (1 - self.activation_graph)
        

class Softmax():
    def __init__(self, layerZGraph, layerPosition=None):
        self.name = "softmax (layer: {0})".format(layerPosition)
    
        """
        Function
        """
        self.activation_graph = th.tensor.exp(layerZGraph) / th.tensor.sum(th.tensor.exp(layerZGraph), dtype="float32")
        
        """
        Derivitive
        """
        self.activationDerv_graph = self.activation_graph * (1 - self.activation_graph)
    

class Linear():
    def __init__(self, layerZGraph, layerPosition=None):
        self.name = "linear (layer: {0})".format(layerPosition)

        self.activation_graph = layerZGraph
        self.activationDerv_graph = 1


def NormalisedFuncDivide(z):
    return z/th.tensor.max(th.tensor.abs_(z))

def NormalisedFuncSubtract(z):
    return z-th.tensor.max(z)


activationFuncs = {"leakyrelu":LeakyRelu,"softmax":Softmax,"sigmoid":Sigmoid, "linear":Linear}
normalisedOps = {"d":NormalisedFuncDivide,"s":NormalisedFuncSubtract}

def AddActivationFunc(aFuncName, layerZGraph, layerPosition=None):
    aFuncNameSplit = aFuncName.split("_")
    if len(aFuncNameSplit) == 1:
        return activationFuncs[aFuncName](layerZGraph, layerPosition=layerPosition)
    else:
        name, ops = (aFuncNameSplit[0], aFuncNameSplit[1])
        for op in ops:
            layerZGraph = normalisedOps[op](layerZGraph)
        return activationFuncs[name](layerZGraph, layerPosition=layerPosition)

