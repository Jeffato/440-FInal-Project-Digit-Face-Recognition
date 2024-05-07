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
nnDigitFolder = "nnDigitFinalWeights2"
save = False

# Neural Network Parameters
inputLayerSize = 784 # 28 x 28 images
hiddenLayerSize = 10 # Arbitrary
outputLayerSize = 10 # Digits 0-9

# Training Parameters
learningRate = 0.08
numEpochs = 500

# Load digits from file
# Outputs a 2D array of size numDigits x imageSize 
def load_digits(file, numDigits):
    images = np.empty(shape = (numDigits, flattenImageDim), dtype=int)
    
    with open(file, 'r') as f:
      for i in range(numDigits):
        data = ''

        for _ in range(image_x_dim):
            data += f.readline().strip("\n")

            # Convert the string of characters to a list of characters - slow and sus, fix later if you have time
            image_flat = np.array([utilFunctions.Integer_Conversion_Function(char) for char in list(data)[:flattenImageDim]])

        images[i] = image_flat
            
    return images

def load_label_Data(file, count):
  labels = np.zeros(shape=count, dtype=(np.int_))

  with open(file, 'r') as f:
    for i in range(count):
        labels[i] = f.readline()

  return labels

def extract_digit_features(image):
    pass

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

    # Initialize values, (Layer 1) weights of input -> hidden and (Layer 2) weights of hidden -> output
    theta1 = np.random.uniform(-0.5, 0.5, (hiddenLayerSize, inputLayerSize))
    bias1 = np.zeros(shape=(hiddenLayerSize, 1), dtype=float)

    theta2 = np.random.uniform(-0.5, 0.5, (outputLayerSize, hiddenLayerSize))
    bias2 = np.zeros(shape=(outputLayerSize, 1), dtype=float)

    error = []

    # Training loop
    for epoch in range(numEpochs):
        # Shuffle Data Set
        indices = np.random.permutation(testsize)
        digits_shuffled = digits[indices]
        digitlabels_shuffled = digitlabels[indices]

        # Convert input to transpose 
        X_T = digits_shuffled.transpose()

        # Forward Prop
        z1, a1, z2, a2 = forward_pass(X_T, theta1, theta2, bias1, bias2)

        # Backwards Prop
        dZ2 = a2 - utilFunctions.one_hot(digitlabels_shuffled.T)
        dW2 = (1 / testsize) * dZ2.dot(a1.T)
        db2 = (1 / testsize) * np.sum(dZ2)

        dZ1 = theta2.T.dot(dZ2) * utilFunctions.ReLU_deriv(z1)
        dW1 = (1 / testsize) * dZ1.dot(X_T.T)
        db1 = (1 / testsize) * np.sum(dZ1)

        # Update weights
        theta1 -= learningRate * dW1
        bias1 -= learningRate * db1
        theta2 -= learningRate * dW2
        bias2 -= learningRate * db2

        # Record Errors
        prediction = np.argmax(a2, 0)

        total_error = 1 - getAccuracy(prediction, digitlabels_shuffled)
        error.append(total_error)

        print(f"Epoch: {epoch}, Total_error: {total_error} ")
    
    endTime = time.time()
    training_time = endTime - startTime

    print("Done!")

    return theta1, theta2, bias1, bias2, training_time, error

# use global var?
def getAccuracy(predictions, label):
    # counter = 0
    # for i in range(label.size):
    #     print(f"Pred: {predictions[i]}, Label: {label[i]}")
    #     if(predictions[i] == label[i]): counter +=1
    
    # print(f"Count: {counter}")

    return np.sum(predictions == label) / label.size

### TESTING CODE ####

digit_train = load_digits("data/digitdata/trainingimages", digitTrainingSize)
digit_train_labels = load_label_Data("data/digitdata/traininglabels", digitTrainingSize)

# # Validation Sets
digit_valid = load_digits("data/digitdata/validationimages", digitValSize)
digit_valid_labels = load_label_Data("data/digitdata/validationlabels", digitValSize)

# # Testing Sets
digit_test = load_digits("data/digitdata/testimages", digitTestSize)
digit_test_labels = load_label_Data("data/digitdata/testlabels", digitTestSize)

theta1, theta2, bias1, bias2, training_time, errors = trainDigits(digit_train, digit_train_labels, digitTrainingSize)

if save: # Training
    # Extract features and labels for the current percentage of data
    # training_features_subset = extract_features(training_images[:num_data_pts])
    # training_labels_subset = training_labels[:num_data_pts]
    
    # Train the model and record time and errors
    theta1, theta2, bias1, bias2, training_time, errors = trainDigits(digit_train, digit_train_labels, digitTrainingSize)
    # Save To File
    if save:
        if not os.path.exists(nnDigitFolder): os.makedirs(nnDigitFolder)

        np.savetxt(os.path.join(nnDigitFolder, f"theta_1{100}_percent.txt"), theta1)
        np.savetxt(os.path.join(nnDigitFolder, f"theta_2{100}_percent.txt"), theta2)
        np.savetxt(os.path.join(nnDigitFolder, f"bias_1{100}_percent.txt"), bias1)
        np.savetxt(os.path.join(nnDigitFolder, f"bias_2{100}_percent.txt"), bias2)
        np.savetxt(os.path.join(nnDigitFolder, f"training_time{100}_percent.txt"), [training_time])
        np.savetxt(os.path.join(nnDigitFolder, f"errors_{100}_percent.txt"), errors)

else: # Reading from file
    theta1 = np.loadtxt(os.path.join(nnDigitFolder, f"theta_1{100}_percent.txt"))
    theta2 = np.loadtxt(os.path.join(nnDigitFolder, f"theta_2{100}_percent.txt"))
    bias1 = np.loadtxt(os.path.join(nnDigitFolder, f"bias_1{100}_percent.txt"))
    bias2 = np.loadtxt(os.path.join(nnDigitFolder, f"bias_2{100}_percent.txt"))
    bias1.shape += (1,)
    bias2.shape += (1,)
    training_time = float(np.loadtxt(os.path.join(nnDigitFolder, f"training_time{100}_percent.txt")))
    errors = np.loadtxt(os.path.join(nnDigitFolder, f"errors_{100}_percent.txt"))

predTest = forward_pass(digit_test.transpose(), theta1, theta2, bias1, bias2)[3]
predictions = np.argmax(predTest, 0)
testAccuracy = getAccuracy(predictions, digit_test_labels)

print(f"Test Accuracy: {testAccuracy} ")