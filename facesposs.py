import numpy as np
import time
import utilFunctions
import os
import math
import sys

# Data Set Size
faceTestSize = 150
faceTrainingSize = 451
faceValSize = 301
image_x_dim = 70
image_y_dim = 60
flattenImageDim = image_x_dim * image_y_dim

# Data Split %
startPercent = 10
endPercent = 101

# Folder locations and Flags
nnFaceFolder = "nnFaceWeights2"
save = True

# Neural Network Parameters
inputLayerSize = 4200 # 70 x 60 images
hiddenLayerSize = 5 # Arbitrary
outputLayerSize = 2 # Face or Not Face

# Training Parameters
learningRate = 0.08
numEpochs = 200

# Load Faces from file
# Outputs a 2D array of size numFaces x imageSize 
def load_faces(file, numFaces):
    images = np.empty(shape = (numFaces, flattenImageDim), dtype=int)
    
    with open(file, 'r') as f:
      for i in range(numFaces):
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
def trainFaces(faces, facelabels, testsize):
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
        faces_shuffled = faces[indices]
        facelabels_shuffled = facelabels[indices]
        
        # Convert input to transpose 
        X_T = faces_shuffled.transpose()

        # Forward Prop
        z1, a1, z2, a2 = forward_pass(X_T, theta1, theta2, bias1, bias2)

        # Backwards Prop
        dZ2 = a2 - utilFunctions.one_hot(facelabels_shuffled.T)
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

        total_error = 1 - getAccuracy(prediction, facelabels_shuffled)
        error.append(total_error)
        print(f"Epoch: {epoch}, Total_error: {total_error} ")
    
    endTime = time.time()
    training_time = endTime - startTime

    print("Done!")

    return theta1, theta2, bias1, bias2, training_time, error

# use global var?
def getAccuracy(predictions, label):
    return np.sum(predictions == label) / label.size

def getAccuracy2(predictions, label):
    counter = 0
    for i in range(label.size):
        print(f"Pred: {predictions[i]}, Label: {label[i]}")
        if(predictions[i] == label[i]): counter +=1
    
    print(f"Count: {counter}, Total: {label.size}")

    return np.sum(predictions == label) / label.size

### TESTING CODE ####

face_train = load_faces("data/facedata/facedatatrain", faceTrainingSize)
face_train_labels = load_label_Data("data/facedata/facedatatrainlabels", faceTrainingSize)

# # Validation Sets
# face_valid = load_faces("data/facedata/facedatavalidation", faceValSize)
# face_valid_labels = load_label_Data("data/facedata/facedatavalidationlabels", faceValSize)

# # Testing Sets
face_test = load_faces("data/facedata/facedatatest", faceTestSize)
face_test_labels = load_label_Data("data/facedata/facedatatestlabels", faceTestSize)

theta1, theta2, bias1, bias2, training_time, errors = trainFaces(face_train, face_train_labels, faceTrainingSize)

predTest = forward_pass(face_test.transpose(), theta1, theta2, bias1, bias2)[3]
predictions = np.argmax(predTest, 0)

testAccuracy = getAccuracy2(predictions, face_test_labels)

print(f"Test Accuracy: {testAccuracy} ")

# if save: # Training
#     # Extract features and labels for the current percentage of data
#     # training_features_subset = extract_features(training_images[:num_data_pts])
#     # training_labels_subset = training_labels[:num_data_pts]
    
#     # Train the model and record time and errors
#     theta1, theta2, bias1, bias2, training_time, errors = trainFaces(face_train, face_train_labels, faceTrainingSize)
#     # Save To File
#     if save:
#         if not os.path.exists(nnFaceFolder): os.makedirs(nnFaceFolder)

#         np.savetxt(os.path.join(nnFaceFolder, f"theta_1{100}_percent.txt"), theta1)
#         np.savetxt(os.path.join(nnFaceFolder, f"theta_2{100}_percent.txt"), theta2)
#         np.savetxt(os.path.join(nnFaceFolder, f"bias_1{100}_percent.txt"), bias1)
#         np.savetxt(os.path.join(nnFaceFolder, f"bias_2{100}_percent.txt"), bias2)
#         np.savetxt(os.path.join(nnFaceFolder, f"training_time{100}_percent.txt"), [training_time])
#         np.savetxt(os.path.join(nnFaceFolder, f"errors_{100}_percent.txt"), errors)

# else: # Reading from file
#     theta1 = np.loadtxt(os.path.join(nnFaceFolder, f"theta_1{100}_percent.txt"))
#     theta2 = np.loadtxt(os.path.join(nnFaceFolder, f"theta_2{100}_percent.txt"))
#     bias1 = np.loadtxt(os.path.join(nnFaceFolder, f"bias_1{100}_percent.txt"))
#     bias2 = np.loadtxt(os.path.join(nnFaceFolder, f"bias_2{100}_percent.txt"))
#     bias1.shape += (1,)
#     bias2.shape += (1,)
#     training_time = float(np.loadtxt(os.path.join(nnFaceFolder, f"training_time{100}_percent.txt")))
#     errors = np.loadtxt(os.path.join(nnFaceFolder, f"errors_{100}_percent.txt"))