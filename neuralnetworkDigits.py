import numpy as np
import time
import utilFunctions
import os
import math
import sys

# Data Set Size
digitTestSize = 1000
digitTrainingSize = 5000
digitValSize = 1000
image_x_dim = 28
image_y_dim = 28
flattenImageDim = image_x_dim * image_y_dim

# Data Split %
startPercent = 10
endPercent = 101

# Folder locations and Flags
pDigitFolder = "nnDigitFinalWeights"
save = True

# Neural Network Parameters
inputLayerSize = 784 # 28 x 28 images
hiddenLayerSize = 10 # Arbitrary
outputLayerSize = 10 # Digits 0-9

# Training Parameters
learningRate = 0.01
numEpochs = 1000

# Load digits from file
# Outputs a 2D array of size numDigits x imageSize 
def load_digits(file, numDigits):
    images = np.empty(shape = (numDigits, flattenImageDim), dtype=(np.unicode_))
    
    with open(file, 'r') as f:
      for i in range(numDigits):
        data = ''

        for _ in range(image_x_dim):
            data += f.readline()
            # Convert the string of characters to a list of characters
            image_flat = list(data)[:flattenImageDim]

        images[i] = image_flat
            
    # Convert ASCII to int
    intConversion =  np.vectorize(utilFunctions.Integer_Conversion_Function) #  Why doesn't this work lol
    # return intConversion(images)
    return images

def load_label_Data(file, count):
  labels = np.zeros(shape=count, dtype=(np.int_))

  with open(file, 'r') as f:
    for i in range(count):
        labels[i] = f.readline()

  return labels

def extract_digit_features(image):
    pass

# Delete later maybe
def forward_pass(features, theta1, theta2, bias1, bias2):
    z1 = theta1.dot(features) + bias1
    a1 = utilFunctions.ReLu(z1)
    z2 = theta2.dot(a1) + bias2
    a2 = utilFunctions.softMax(z2)
    
    return (z1, a1, z2, a2)

# Takes in 2D array of digits (numDigits x 784), 
def trainDigits(digits, digitlabels, testsize):
    # Record start time
    startTime = time.time()

    # Convert input to transpose 
    X_T = digits.transpose

    # Initialize values, (Layer 1) weights of input -> hidden and (Layer 2) weights of hidden -> output
    theta1 = np.random.uniform(-1, 1, (hiddenLayerSize, inputLayerSize))
    bias1 = np.zeros(shape=(hiddenLayerSize, 1), dtype=float)

    theta2 = np.random.uniform(-1, 1, (outputLayerSize, hiddenLayerSize))
    bias2 = np.zeros(shape=(outputLayerSize, 1), dtype=float)

    error = []

    # Training loop
    for epoch in range(numEpochs):
        # Shuffle Data Set
        indices = np.random.permutation(testsize)
        digits_shuffled = X_T[indices]
        digitlabels_shuffled = digitlabels[indices]

        # Forward Prop
        z1 = theta1.dot(digits_shuffled) + bias1
        a1 = utilFunctions.ReLu(z1)
        z2 = theta2.dot(a1) + bias2
        a2 = utilFunctions.softMax(z2)

        # Backwards Prop
        


### TESTING CODE ####

np.set_printoptions(threshold=sys.maxsize)
digit_train = load_digits("data/digitdata/trainingimages", digitTrainingSize)
digit_train_labels = load_label_Data("data/digitdata/traininglabels", digitTrainingSize)



# #Training Sets
# digit_train = utilFunctions.load_Image_Data("data/digitdata/trainingimages", digitTrainingSize, 28, 28)
# digit_train_labels = utilFunctions.load_label_Data("data/digitdata/traininglabels", digitTrainingSize)

# # Validation Sets
# digit_valid = utilFunctions.load_Image_Data("data/digitdata/validationimages", digitValSize, 28, 28)
# digit_valid_labels = utilFunctions.load_label_Data("data/digitdata/validationlabels", digitValSize)

# # Testing Sets
# digit_test = utilFunctions.load_Image_Data("data/digitdata/testimages", digitTestSize, 28, 28)
# digit_test_labels = utilFunctions.load_label_Data("data/digitdata/testlabels", digitTestSize)

