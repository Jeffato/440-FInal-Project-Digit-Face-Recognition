import numpy as np

'''
Pixel Values
  0: no edge (blank)
  1: gray pixel (+) [used for digits only]
  2: edge [for face] or black pixel [for digit] (#)
'''

# Converts Ascii input to numerical representation
def Integer_Conversion_Function(character):
  if(character == ' '):
    return int(0)
  elif(character == '+'):
    return int(1)
  elif(character == '#'):
    return int(2)

# Loads count number of images of size (x_dim, y_dim) from file
# Returns 3-D numpy array (number of image, x_pixel, y_pixel)
def load_Image_Data(file, count, x_dim, y_dim):
  images = np.empty(shape = (count, x_dim, y_dim), dtype=(np.unicode_))

  # Open file. Read each line into numpy array
  with open(file, 'r') as f:
    for i in range(count):
        for j in range(x_dim):
          images[i][j] = [char for char in f.readline()][0:y_dim]

  # Function to convert ascii to integer
  intConvertVector = np.vectorize(Integer_Conversion_Function)

  # return intConvertVector(images)
  return images

# Loads count number of labels from file
# Return 1-D numpy array of labels
def load_label_Data(file, count):
  labels = np.zeros(shape=count, dtype=(np.int_))

  with open(file, 'r') as f:
    for i in range(count):
        labels[i] = f.readline()

  return labels

# binary activation
def binary_Activation(x):
  if x >= 0: return 1
  else: return 0

def ReLu(x):
  return np.maximum(0, x)

def ReLU_deriv(x):
  return x > 0

def softMax(x):
  return np.exp(x) / sum(np.exp(x))

def one_hot(X):
    one_hot_Y = np.zeros((X.size, X.max() + 1))
    one_hot_Y[np.arange(X.size), X] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
