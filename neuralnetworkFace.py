import numpy as np
import time
import utilFunctions
import os
import math

# Constants
numFeatures = 20
numEpochs = 1000
learningRate = 1
regularization_constant = 0.01

hidden_nodes = 10
output_nodes = 2
theta1_size = hidden_nodes * numFeatures
theta2_size = output_nodes * hidden_nodes # Check to add theta1

# Data Set Size
faceTestSize = 150
faceTrainingSize = 451
faceValSize = 301

# Data Split %
startPercent = 10
endPercent = 101

# Folder locations and Flags
pFaceFolder = "nnFaceFinalWeights"
save = False

# Face features -> number of pixels in a segment of the picture
def extract_Face_Feature(image):
    reshaped_image = image.reshape(numFeatures, 14, 15)
    counts = np.array([np.count_nonzero(slice) for slice in reshaped_image])

    return counts

def forward_pass(features, theta1, theta2, bias1, bias2):
    a2 = utilFunctions.sigmoid(np.dot(features, theta1) + bias1)
    a3 = utilFunctions.sigmoid(np.dot(a2, theta2) + bias2)

    return (a2, a3)

# Calculates the cost error J_theta
def j_theta_loss(training_labels, n, h_theta, theta1, theta2):
    s_l = [numFeatures, hidden_nodes, output_nodes] 

    j_theta = (-1/n) * (
            sum(
                sum(
                    training_labels[i][k]*math.log(h_theta[i][k][0])+(1-training_labels[i][k])*math.log(1-h_theta[i][k][0])
                for k in range(0, output_nodes) # For result element k in training sample i
                )
            for i in range(0,n) # For n training samples
            )
        ) + (regularization_constant / (2 * n)) * sum(
            sum (
                sum(
                    pow(theta1[j,i],2) if l == 1 else pow(theta2[j,i],2)
                for j in range(0, s_l[l]) # For nodes in current layer (no bias node in matrix so start is still 0)    
                )
            for i in range(0, s_l[l-1]) # For nodes in previous layer (no bias node in matrix so start is still 0)
            )
        for l in range(1,3) # For layers 2,3 (hidden, output)
        )
    return j_theta

def trainFace(faces, facelabels, testsize):
    # Record start time
    startTime = time.time()

    # Initialize values, (1) weights of input -> hidden and (2) weights of hidden -> output
    theta1 = np.random.uniform(-1, 1, (numFeatures, hidden_nodes))
    theta2 = np.random.uniform(-1, 1, (hidden_nodes, output_nodes))

    bias1 = np.zeros(shape=(1, hidden_nodes), dtype=float)
    bias2 = np.zeros(shape=(1, output_nodes), dtype=float)

    errors = []

    # Training loop
    for epoch in range(numEpochs):
        # Shuffle indices for each epoch
        indices = np.random.permutation(testsize)
        faces_shuffled = faces[indices]
        facelabels_shuffled = facelabels[indices]

        # Initialize output and gradients
        predicted_output = np.empty(shape=(testsize, 1, output_nodes), dtype=float)
        Delta_l1 = np.zeros(shape=(hidden_nodes,numFeatures))
        Delta_l2 = np.zeros(shape=(output_nodes, hidden_nodes))
        
        for i in range(testsize):
            features = extract_Face_Feature(faces_shuffled[i]).reshape(1,-1)
            label = facelabels_shuffled[i]
            
            # Forward Propagation
            (a2, predicted_output[i]) = forward_pass(features, theta1, theta2, bias1, bias2)

            # Backwards Propagation
            delta_l3 = predicted_output[i] - label
            delta_l2 = np.dot(a2, (theta2 * delta_l3))  * (np.ones(shape=(1, hidden_nodes)) - a2)

            # Compute Gradients
            Delta_l1 += (delta_l2 * features)
            Delta_l2 += (delta_l3 * a2)
        
        # Compute Avg Regularized Gradient
        l1_reg = 1/testsize * Delta_l1 + regularization_constant * theta1
        l2_reg = 1/testsize * Delta_l2 + regularization_constant * theta2
        
        # # Gradient Checking
        # DVec = np.concatenate((D_l1.flatten(), D_l2.flatten()),dtype=float)
        # c = 1E-4
        # gradApprox = np.empty(shape=(theta.shape[0]), dtype=float)
        # for i in range(0, 5):#gradApprox.shape[0]):
        #     theta[i] += c
        #     temp1 = cost_error(training_labels, n, h_theta, theta)
        #     theta[i] -= 2*c
        #     temp2 = cost_error(training_labels, n, h_theta, theta)
        #     theta[i] += c
        #     gradApprox[i] = (temp1 - temp2)/(2 * c)
        # # print(abs(gradApprox-DVec))

        # Update Weights and Biases
        theta1 -= learningRate * l1_reg
        theta2 -= learningRate * l2_reg
        bias1 -= learningRate * (np.sum(delta_l2, axis=1, keepdims=True) / testsize)
        bias2 -= learningRate * (np.sum(delta_l3, axis=1, keepdims=True) / testsize)
        
        # Record Errors
        total_error = np.sum(predicted_output != facelabels_shuffled) / testsize
        errors.append(total_error)

        print(f"Epoch: {epoch}, Total_error: {np.sum(total_error)} ")

    endTime = time.time() # Record end time
    training_time = endTime - startTime # Calculate training time

    print("Training complete!\n")
    return theta1, theta2, bias1, bias2, training_time, errors


### TESTING ###

#Training Sets
face_train = utilFunctions.load_Image_Data("data/facedata/facedatatrain", faceTrainingSize, 70, 60)
face_train_labels = utilFunctions.load_label_Data("data/facedata/facedatatrainlabels", faceTrainingSize)

# Validation Sets
face_valid = utilFunctions.load_Image_Data("data/facedata/facedatavalidation", faceValSize, 70, 60)
face_valid_labels = utilFunctions.load_label_Data("data/facedata/facedatavalidationlabels", faceValSize)

# Testing Sets
face_test = utilFunctions.load_Image_Data("data/facedata/facedatatest", faceTestSize, 70, 60)
face_test_labels = utilFunctions.load_label_Data("data/facedata/facedatatestlabels", faceTestSize)

trainingTimes = []
errorSet = {}

theta1, theta2, bias1, bias2, training_time, errors = trainFace(face_train, face_train_labels, 10)

# for percentage in range(startPercent, endPercent, 10):
#     numDataPts = int(faceTrainingSize * percentage / 100)

#     if save:
#         # Create the folder if it doesn't exist
#         if not os.path.exists(pFaceFolder): os.makedirs(pFaceFolder)

#         print(f"Training with {percentage}% of the training data")

#         # Train model and save results to file
#         theta1, theta2, bias1, bias2, training_time, errors = trainFace(face_train, face_train_labels, numDataPts)
#         np.savetxt(os.path.join(pFaceFolder, f"theta_1_{percentage}.txt"), theta1)
#         np.savetxt(os.path.join(pFaceFolder, f"theta_2_{percentage}.txt"), theta2)
#         np.savetxt(os.path.join(pFaceFolder, f"bias_1_{percentage}.txt"), [bias1])
#         np.savetxt(os.path.join(pFaceFolder, f"bias_2_{percentage}.txt"), [bias2])
#         np.savetxt(os.path.join(pFaceFolder, f"training_time_{percentage}.txt"), [training_time])
#         np.savetxt(os.path.join(pFaceFolder, f"errors_{percentage}.txt"), errors)

#     else:
#         theta1 = np.loadtxt(os.path.join(pFaceFolder, f"theta_1_{percentage}.txt"))
#         theta2= np.loadtxt(os.path.join(pFaceFolder, f"theta_2_{percentage}.txt"))
#         bias1 = np.loadtxt(os.path.join(pFaceFolder, f"bias_1_{percentage}.txt"))
#         bias2 = np.loadtxt(os.path.join(pFaceFolder, f"bias_2_{percentage}.txt"))
#         faceTrainingTime = float(np.loadtxt(os.path.join(pFaceFolder, f"training_time_{percentage}.txt")))
#         errors = np.loadtxt(os.path.join(pFaceFolder, f"errors_{percentage}.txt"))
    