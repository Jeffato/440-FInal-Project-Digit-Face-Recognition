import numpy as np
import time
import utilFunctions
import os

# Constants
numFeatures = 20
numEpochs = 10000
learningRate = 1

# Data Set Size
faceTestSize = 150
faceTrainingSize = 451
faceValSize = 301

# Data Split %
startPercent = 10
endPercent = 101