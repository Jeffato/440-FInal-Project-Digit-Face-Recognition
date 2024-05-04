import numpy as np
import time
import utilFunctions

# Constants
numFeatures = 20
numEpochs = 10000
learningRate = 1

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

def forwardPass(features, weights, bias):
    # Compute the weighted sum of inputs
    weightedSum = np.dot(features, weights) + bias
    
    # Apply activation function (sigmoid)
    output = utilFunctions.binary_Activation(weightedSum)
    return output

# Face features -> number of pixels in a segment of the picture
def extract_Face_Feature(image):
    reshaped_image = image.reshape(20, 14, 15)
    counts = [np.count_nonzero(slice) for slice in reshaped_image]

    return counts

def trainFace(faces, facelabels, testsize):
    startTime = time.time()

    # Initialize weights
    weights = np.random.uniform(-1, 1, numFeatures)
    bias = 0

    errors = []

    # Loop through testing set
    for epoch in numEpochs:
        # Shuffle data set
        indices = np.random.permutation(testsize)
        faces_shuffled = faces[indices]
        facelabels_shuffled = facelabels[indices]

        # Loop through images 
        for i in range(len(testsize)):
            # Extract features
            features = extract_Face_Feature(faces_shuffled[i])
            incorrect = True

            while incorrect:
                predicted_result = forwardPass(features, weights, bias)

                # Compute Error
                # If error is positive -> predicted face, not face
                # If error is negative -> predicted not face, face
                error = facelabels_shuffled[i] - predicted_result

                if error == 0:
                    incorrect = False

                # Update weights
                weights += learningRate * np.dot(features.T, error)
                bias += learningRate * np.sum(error, axis=0)
    
        # Record error at the end of each epoch
        epochPrediction = []

        for face in faces_shuffled:
            epochPrediction.append(forwardPass(face, weights, bias))
        
        total_error = facelabels - epochPrediction
        errors.append(np.mean(np.abs(total_error)))

        # Print loss (Mean Squared Error) every 1000 epochs
        if epoch % 1000 == 0:
            loss = np.mean(np.square(total_error))
            print(f'Epoch {epoch}: Loss = {loss}')

    endTime = time.time()  # Record end time
    trainingTime = endTime - startTime  # Calculate training time

    print("Training complete!\n")

    return weights, bias, trainingTime, errors

    
    # Training the model

# test = utilFunctions.load_Image_Data("data/facedata/facedatatest", 150, 70, 60)[0]
# print(test.shape)
# print(extract_Face_Feature(test))

