import time
import collections

import dill as pickle
import theano as th
import numpy as np

from lib import activation
from lib import timeStuff



"""
This network model does not have a representation for the input layer, we just apply the inputs
to the first hidden layer
"""
class NetworkCreator():
    @timeStuff.timeIt("Compilation")
    def __init__(self, networkDesign, ParamGen, Cost, rValue, momentumValue):
        """
        Network vars
        """
        self.networkDesign = networkDesign.lower()  
        self.rValue = np.float32(rValue)
        self.momentumValue = np.float32(momentumValue)
        self.Cost = Cost()

        "Network initialisation"
        #  Layer storage
        self.networkStructure = []

        #  Layer info in iterable format
        layerSplit = [[int(layer.split(",")[0]),layer.split(",")[1]] for layer in self.networkDesign.split("|")]
        #  Iterate throught the hidden layer
        for layerNumber, (prevLayer, currentLayer, nextLayer) in enumerate(zip(layerSplit[:-2], layerSplit[1:-1], layerSplit[2:])):
            self.networkStructure.append(Layer(ParamGen, currentLayer[1], self.rValue, self.momentumValue, layerPosition=layerNumber, prevLayerSize=prevLayer[0], currentLayerSize=currentLayer[0], nextLayerSize=nextLayer[0]))

        #  The output layer
        self.networkStructure.append(Layer(ParamGen, layerSplit[-1][1], self.rValue, self.momentumValue, Cost, prevLayerSize=layerSplit[-2][0], currentLayerSize=layerSplit[-1][0], nextLayerSize=1))


        """
        Forward prop
        """
        x_initial = th.tensor.fvector(name="ForwardProp_x")
        self.networkStructure[0].forward(x_initial)
        x = self.networkStructure[0].y
        for layer in self.networkStructure[1:]:
            layer.forward(x)
            #  The output of our layer is the input for the next layer
            x = layer.y
            
        self.__ForwardPropogate__ = th.function(inputs=[x_initial], outputs=self.networkStructure[-1].y)
            

        """
        Backward prop
        """
        desire = th.tensor.fvector(name="BackwardProp_desire")
        if self.Cost.name == "crossentropy" and (self.networkStructure[-1].A.name.split()[0] == "softmax" or self.networkStructure[-1].A.name.split()[0] == "sigmoid"):
            errorWz = self.networkStructure[-1].y - desire
        else:
            self.Cost.calc(self.networkStructure[-1].y, desire)
            errorWz = self.networkStructure[-1].A.activationDerv_graph * self.Cost.costDerv_graph

        allUpdates = []
        errorWy, updates = self.networkStructure[-1].train(errorWz)
        allUpdates.extend(updates)
        for layer_reversed in self.networkStructure[::-1][1:]:
            errorWz = layer_reversed.y * errorWy
            errorWy, updates = layer_reversed.train(errorWz)
            allUpdates.extend(updates)
    
        self.__BackwardPropogate__ = th.function(inputs=[x_initial, desire], outputs=self.networkStructure[0].errorWy_new, updates=allUpdates)

    @timeStuff.timeIt("Training")
    def Propogate(self, inputData, desiredData):
        for x, desire in zip(inputData, desiredData):
            self.__BackwardPropogate__(x, desire)


    def Predict(self, x):
        return np.argmax(self.__ForwardPropogate__(x))

    
    def Save(self, file2save2):
        with open(file2save2,"wb") as file2save2:
            pickle.dump(self, file2save2)



class Layer():
    def __init__(self, ParamGen, activation, rValue, momentumValue, layerPosition=None, **kwargs):
        """
        Layer propoerties
        """
        self.layerPosition = layerPosition
        self.w_shared, self.b_shared = ParamGen.GenParams(layerPosition, **kwargs)
        self.activationName = activation
        self.rValue = rValue
        self.momentumValue = momentumValue
        self.wMomentum_shared = th.shared(np.zeros(shape=self.w_shared.get_value().shape, dtype="float32"), name="momentum_weight (layer {0})".format(layerPosition), borrow=True)
        self.bMomentum_shared = th.shared(np.zeros(shape=self.b_shared.get_value().shape, dtype="float32"), name="momentum_bias (layer {0})".format(layerPosition), borrow=True)
       
        
    def forward(self, x):
        self.x = x
        z = th.tensor.dot(x, self.w_shared.transpose()) + self.b_shared 
        self.A = activation.AddActivationFunc(self.activationName, z, self.layerPosition)
        self.y = self.A.activation_graph
      

    def train(self, errorWz):
        errorWw = (self.momentumValue * self.wMomentum_shared) + (self.rValue * (th.tensor.extra_ops.repeat(errorWz.reshape((1,-1)).transpose(), self.x.shape[0], axis=1) * th.tensor.extra_ops.repeat(self.x.reshape((1,-1)), errorWz.shape[0], axis=0)))
        errorWb = (self.momentumValue * self.bMomentum_shared) + (self.rValue * errorWz)
        
        """
        Update vars, update momentum
        """
        updates = (
                (self.w_shared, self.w_shared - errorWw),
                (self.b_shared, self.b_shared - errorWb),
                (self.wMomentum_shared, errorWw),
                (self.bMomentum_shared, errorWb)
                )

        self.errorWy_new = th.dot(errorWz, self.w_shared)
        return self.errorWy_new, updates


    



