import os
import numpy as np
import matplotlib.pyplot as plt

# folders to read the data from
# folders = ["pDigitFinalWeights", "pFaceFinalWeights"]
folders = ["nnFaceFinalWeights", "nnDigitFinalWeights"]

# data points from 0 to 100, increasing by 10
dataPointPercent = np.arange(0, 101, 10, dtype=int)

# dictionary to store the data for each folder
trainingTimesDict = {}
predictionErrorDict = {}
stdDeviationDict = {}

# Loop through each folder
for folder in folders:
    trainingTimes = [0]  # Start with 0 for 0% training time
    predictionErrors = [0]
    stdDeviations = [0]
    
    for percentage in range(10, 101, 10):
        # training times
        filename = f"training_time_{percentage}.txt"
        filePath = os.path.join(folder, filename)
        time = float(np.loadtxt(filePath))
        trainingTimes.append(time)

        # prediction errors
        filename2 = f"errors_{percentage}.txt"
        filepath2 = os.path.join(folder, filename2)
        error = (np.loadtxt(filepath2))
        predictionErrors.append(np.mean(error))

        # standard deviation
        stdDeviations.append(np.std(error))
    
    # Store the data array in the dictionary with the folder name as the key
    trainingTimesDict[folder] = trainingTimes
    predictionErrorDict[folder] = predictionErrors
    stdDeviationDict[folder] = stdDeviations


# Plot the training times against data point percent for all
plt.figure(figsize=(10, 6))
for folderName, trainingTimes in trainingTimesDict.items():
    plt.scatter(dataPointPercent, trainingTimes, label=folderName)

plt.xlabel('Percentage of Data Points')
plt.ylabel('Training Time')
plt.title('Training Time vs Data Points')
plt.grid(True, axis="both")
plt.legend()
plt.show()

# Plot the prediction errors against data point percent each other for all
plt.figure(figsize=(10, 6))
for folderName, predictionErrors in predictionErrorDict.items():
    plt.scatter(dataPointPercent, predictionErrors, label=folderName)

plt.xlabel('Percentage of Data Points')
plt.ylabel('Training Prediction Error')
plt.title('Error vs Data Points')
plt.grid(True, axis="both")
plt.legend()
plt.show()

# Plot the standard deviations against data point percent each other for all
plt.figure(figsize=(10, 6))
for folderName, stdDeviations in stdDeviationDict.items():
    plt.scatter(dataPointPercent, stdDeviations, label=folderName)

plt.xlabel('Percentage of Data Points')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Data Points')
plt.grid(True, axis="both")
plt.legend()
plt.show()