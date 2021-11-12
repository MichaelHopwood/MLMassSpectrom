import numpy as np
from core import NN, VisualizeLayers

nn = NN()
nn.make_model()
nn.train()
nn.evaluate()

vz = VisualizeLayers()
vz.visualize(nn.model)
