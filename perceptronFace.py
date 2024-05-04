import numpy as np
import time
import utilFunctions
import os
'''
Perceptron:

Perceptron is a linear classifier that takes in features, weighs them, then sums them for an output.

Single Class- Faces
F(X) = sum [ weight * feature(X) ] + bias

F(X) >= 0 -> face
F(X) < 0  -> not face

Psuedo:
Intialize weights randomly or 0

For each training image:
    Predict using F(X)
        If correct move on
        
        If predict face, label not face
            subtract feature from weight
            subtract one from bias

        If predict not face, label face
            add feature to weight
            add one from bias

Continue until
	- Values converge
	- Time limit exceeded
    - All images identified correctly in 1 cycle
'''
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

# Folder locations and Flags
pFaceFolder = "pFaceFinalWeights"
save = False

def forwardPass(features, weights, bias):
    # Compute the weighted sum of inputs
    weightedSum = np.dot(features, weights) + bias
    
    # Apply activation function 
    output = utilFunctions.binary_Activation(weightedSum)
    return output

# Face features -> number of pixels in a segment of the picture
def extract_Face_Feature(image):
    x_dim = numFeatures
    y_dim = numFeatures

    reshaped_image = image.reshape(numFeatures, 14, 15)
    counts = np.array([np.count_nonzero(slice) for slice in reshaped_image])

    return counts

def trainFace(faces, facelabels, testsize):
    startTime = time.time()

    # Initialize weights
    weights = np.random.uniform(-1, 1, numFeatures)
    bias = 0

    errors = []

    # Loop through testing set
    for epoch in range(numEpochs+1):
        # Shuffle data set
        indices = np.random.permutation(testsize)
        faces_shuffled = faces[indices]
        facelabels_shuffled = facelabels[indices]

        # Loop through images 
        for i in range(testsize):
            # Extract features
            features = extract_Face_Feature(faces_shuffled[i])
            predicted_result = forwardPass(features, weights, bias)

            # Compute Error
            # If error is positive -> predicted face, not face
            # If error is negative -> predicted not face, face
            error = facelabels_shuffled[i] - predicted_result

            # Update weights
            weights += learningRate * np.dot(features, error)
            bias += learningRate * error
    
        # Record error at the end of each epoch
        epochPrediction = []

        for face in faces_shuffled:
            feature = extract_Face_Feature(face)
            prediction = forwardPass(feature, weights, bias)
            epochPrediction.append(prediction)
        
        total_error = facelabels_shuffled[0:testsize] - epochPrediction
        errors.append(np.mean(np.abs(total_error)))

        # Print loss (Mean Squared Error) every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(total_error))
            print(f'Epoch {epoch}: Loss = {loss}')

    endTime = time.time()  # Record end time
    trainingTime = endTime - startTime  # Calculate training time

    print("Training complete!\n")

    return weights, bias, trainingTime, errors

def evalModel(image, labels, weights, bias):
    predictedLabels = []

    for face in image:
        feature = extract_Face_Feature(face)
        prediction = forwardPass(feature, weights, bias)
        predictedLabels.append(prediction)

    # Compute accuracy
    accuracy = np.mean(predictedLabels == labels)
    
    return accuracy

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

# Iterate using different test sizes 10 20 ... 100
for percentage in range(startPercent, endPercent, 10):
    numDataPts = int(faceTrainingSize * percentage / 100)

    if save:
        # Create the folder if it doesn't exist
        if not os.path.exists(pFaceFolder): os.makedirs(pFaceFolder)

        print(f"Training with {percentage}% of the training data")

        # Train model and save results to file
        faceWeights, faceBias, faceTrainingTime, errors = trainFace(face_train, face_train_labels, numDataPts)
        np.savetxt(os.path.join(pFaceFolder, f"theta_{percentage}.txt"), faceWeights)
        np.savetxt(os.path.join(pFaceFolder, f"bias_{percentage}.txt"), [faceBias])
        np.savetxt(os.path.join(pFaceFolder, f"training_time_{percentage}.txt"), [faceTrainingTime])
        np.savetxt(os.path.join(pFaceFolder, f"errors_{percentage}.txt"), errors)

    else:
        faceWeights = np.loadtxt(os.path.join(pFaceFolder, f"theta_{percentage}.txt"))
        faceBias = np.loadtxt(os.path.join(pFaceFolder, f"bias_{percentage}.txt"))
        faceTrainingTime = float(np.loadtxt(os.path.join(pFaceFolder, f"training_time_{percentage}.txt")))
        errors = np.loadtxt(os.path.join(pFaceFolder, f"errors_{percentage}.txt"))
    
    # Store training time
    trainingTimes.append(faceTrainingTime)
    
    # Store errors for this data size
    errorSet[numDataPts] = errors
    
    # Evaluate the model on validation data
    validationAcc = evalModel(face_valid, face_valid_labels, faceWeights, faceBias)
    print("%19s: %4.2f%%" % ("Validation Accuracy",validationAcc * 100))

    testAcc = evalModel(face_test, face_test_labels, faceWeights, faceBias)
    print("%19s: %4.2f%%\n" % ("Test Accuracy" ,testAcc * 100))

# Compute mean and standard deviation of errors for each data size
meanErrorsBySize = {size: np.mean(errors) for size, errors in errorSet.items()}
stdErrorsBySize = {size: np.std(errors) for size, errors in errorSet.items()}

# Print statistics
print("Mean Errors by Data Size\n------------------------")
for mean in meanErrorsBySize:
    print("n=%4d: %.14f" % (mean, meanErrorsBySize[mean]))
print("\nStandard Deviations of Errors by Data Size\n-------------------------------------------")
for std in stdErrorsBySize:
    print("n=%4d: %.14f" % (std, stdErrorsBySize[std]))
print("\nTraining Times\n--------------")
for train in range(len(trainingTimes)):
    print("n=%4d: %.14f" % ((train + 1) * 500, trainingTimes[train]))