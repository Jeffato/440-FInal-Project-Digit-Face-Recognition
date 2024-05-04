import numpy as np
import time
import utilFunctions
import os
'''
See Face for single class Perceptron:

Digits (Multi Class) Perceptron functions similarly

Each class (digit) has its own function and own weights
Predicted digit -> argmax activation of each digit function

For incorrect guesses:
    Reduce the incorrectly guessed class
    Increase the correct labels class

Digits: 28 x 28 - End with empty line
'''

# Constants
numFeatures = 16
numEpochs = 10000
numClasses = 10 # Digits can be 0 to 9
learningRate = 1

# Data Set Size
digitTestSize = 1000
digitTrainingSize = 5000
digitValSize = 1000

# Data Split %
startPercent = 10
endPercent = 101

# Folder locations and Flags
pDigitFolder = "pDigitFinalWeights"
save = False

def forwardPass(features, weights, bias):
    # Compute the weighted sum of inputs
    weightedSum = np.dot(features, weights) + bias
    
    # Apply activation function 
    output = np.argmax(weightedSum)
    return output

# Digit features -> number of pixels in a segment of the picture 
# Split into 16 features of size (7,7)
def extract_Digit_Feature(image):
    reshaped_image = image.reshape(numFeatures, 7, 7)
    counts = np.array([np.count_nonzero(slice) for slice in reshaped_image])

    return counts

def trainDigit(digits, digitlabels, testsize):
    startTime = time.time()

    # Initialize weights
    weights = np.random.uniform(-1, 1, size = (numFeatures, numClasses))
    bias = np.random.randn(numClasses)

    errors = []

    # Loop through testing set
    for epoch in range(numEpochs+1):
        # Shuffle data set
        indices = np.random.permutation(testsize) # Check
        digits_shuffled = digits[indices]
        digitlabels_shuffled = digitlabels[indices]

        # Loop through images 
        for i in range(testsize):
            # Extract features
            features = extract_Digit_Feature(digits_shuffled[i])
            predicted_result = forwardPass(features, weights, bias)

            # Compute Error
            error = digitlabels_shuffled[i] - predicted_result

            # Update weights
            weights += learningRate * np.dot(features.T, error)
            bias += learningRate * np.sum(error, axis=0)
    
        # Record error at the end of each epoch
        epochPrediction = []

        for digit in digits_shuffled:
            feature = extract_Digit_Feature(digit)
            prediction = forwardPass(feature, weights, bias)
            epochPrediction.append(prediction)
        
        total_error = digitlabels_shuffled[0:testsize] - epochPrediction
        errors.append(np.mean(np.abs(total_error)))

        # Print loss (Mean Squared Error) every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(total_error))
            print(f'Epoch {epoch}: Loss = {loss}')

    endTime = time.time()  # Record end time
    trainingTime = endTime - startTime  # Calculate training time

    print("Done!\n")

    return weights, bias, trainingTime, errors

def evalModel(image, labels, weights, bias):
    predict = []

    for face in image:
        feature = extract_Digit_Feature(face)
        prediction = forwardPass(feature, weights, bias)
        predict.append(prediction)

    # Convert predicted probabilities to class labels
    predictedLabels = np.argmax(predict, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(predictedLabels == labels)
    
    return accuracy

### TESTING ###

#Training Sets
digit_train = utilFunctions.load_Image_Data("data/facedata/facedatatrain", digitTrainingSize, 28, 28)
digit_train_labels = utilFunctions.load_label_Data("data/facedata/facedatatrainlabels", digitTrainingSize)

# Validation Sets
digit_valid = utilFunctions.load_Image_Data("data/facedata/facedatavalidation", digitValSize, 28, 28)
digit_valid_labels = utilFunctions.load_label_Data("data/facedata/facedatavalidationlabels", digitValSize)

# Testing Sets
digit_test = utilFunctions.load_Image_Data("data/facedata/facedatatest", digitTestSize, 28, 28)
digit_test_labels = utilFunctions.load_label_Data("data/facedata/facedatatestlabels", digitTestSize)

trainingTimes = []
errorSet = {}

# Iterate using different test sizes 10 20 ... 100
for percentage in range(startPercent, endPercent, 10):
    numDataPts = int(digitTrainingSize * percentage / 100)

    if save:
        # Create the folder if it doesn't exist
        if not os.path.exists(pDigitFolder): os.makedirs(pDigitFolder)

        print(f"Training with {percentage}% of the training data")

        # Train model and save results to file
        faceWeights, faceBias, faceTrainingTime, errors = trainDigit(digit_train, digit_train_labels, numDataPts)
        np.savetxt(os.path.join(pDigitFolder, f"theta_{percentage}.txt"), faceWeights)
        np.savetxt(os.path.join(pDigitFolder, f"bias_{percentage}.txt"), [faceBias])
        np.savetxt(os.path.join(pDigitFolder, f"training_time_{percentage}.txt"), [faceTrainingTime])
        np.savetxt(os.path.join(pDigitFolder, f"errors_{percentage}.txt"), errors)

    else:
        faceWeights = np.loadtxt(os.path.join(pDigitFolder, f"theta_{percentage}.txt"))
        faceBias = np.loadtxt(os.path.join(pDigitFolder, f"bias_{percentage}.txt"))
        faceTrainingTime = float(np.loadtxt(os.path.join(pDigitFolder, f"training_time_{percentage}.txt")))
        errors = np.loadtxt(os.path.join(pDigitFolder, f"errors_{percentage}.txt"))
    
    # Store training time
    trainingTimes.append(faceTrainingTime)
    
    # Store errors for this data size
    errorSet[numDataPts] = errors
    
    # Evaluate the model on validation data
    validationAcc = evalModel(digit_valid, digit_valid_labels, faceWeights, faceBias)
    print("%19s: %4.2f%%" % ("Validation Accuracy",validationAcc * 100))

    testAcc = evalModel(digit_test, digit_test_labels, faceWeights, faceBias)
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