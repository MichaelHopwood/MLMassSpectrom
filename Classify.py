import numpy as np
from core import NN, VisualizeRNNLayers

nn = NN()
nn.make_model()
nn.train()
nn.evaluate()

vz = VisualizeRNNLayers()
vz.visualize(nn.model)
