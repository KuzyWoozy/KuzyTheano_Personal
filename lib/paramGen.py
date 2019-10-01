import theano as th
import numpy as np


class NormalisedParam():
    def __init__(self):
        self.name = "Number of parameters method"

    def GenParams(self, layerPosition=None, **kwargs):
        standardDeviation = np.sqrt(1/(kwargs["prevLayerSize"] + 1))
        bias_shared = th.shared(np.random.normal(scale=standardDeviation, size=(kwargs["currentLayerSize"])).astype("float32"), borrow=True, name="b (layer: {0})".format(layerPosition))
        weight_shared = th.shared(np.random.normal(scale=standardDeviation, size=(kwargs["currentLayerSize"], kwargs["prevLayerSize"])).astype("float32"), borrow=True, name="w (layer: {0})".format(layerPosition))
        return weight_shared, bias_shared


paramGenerators = {"normal": NormalisedParam}