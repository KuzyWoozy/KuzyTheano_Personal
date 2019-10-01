import theano as th
import numpy as np


class CrossEntropy():
    """
    Gradient clipped cross entropy loss
    """
    def __init__(self):
        self.name = "crossentropy"
        #  The gradient clip for the error
        self.gradientClip = 100
        
    def calc(self, y, desire):
        """
        Cross entropy
        """
        #  Find the special cases to be handled seperately
        wantedOneGotZero = th.tensor.bitwise_and(th.tensor.eq(desire, 1), th.tensor.eq(y, 0))
        wantedZeroGotOne = th.tensor.bitwise_and(th.tensor.eq(desire, 0), th.tensor.eq(y, 1))

        #  Clip the gradients
        func = th.tensor.set_subtensor(y[wantedOneGotZero], self.gradientClip)
        func = th.tensor.set_subtensor(func[wantedZeroGotOne], self.gradientClip)
        
        #  Carry out the Cross Entropy calculation
        func = th.tensor.set_subtensor(y[th.tensor.bitwise_and(th.tensor.eq(desire,0), th.tensor.invert(wantedZeroGotOne))], -(th.tensor.log(1-func[th.tensor.bitwise_and(th.tensor.eq(desire,0), th.tensor.invert(wantedZeroGotOne))])))
        self.cost_graph = th.tensor.set_subtensor(y[th.tensor.bitwise_and(th.tensor.eq(desire,1), th.tensor.invert(wantedOneGotZero))], -(th.tensor.log(func[th.tensor.bitwise_and(th.tensor.eq(desire,1), th.tensor.invert(wantedOneGotZero))])))

        """
        Cross entropy derivitive
        """
        self.costDerv_graph = (y-desire)/y*(1-y)

class MSE:
    def __init__(self):
        self.name = "Mean squared error"

    def calc(self, y, desire):

        self.cost_graph = (y - desire) ** 2
        self.costDerv_graph = 2 * (y - desire)

 
costFuncs = {"crossentropy": CrossEntropy, "mse": MSE}